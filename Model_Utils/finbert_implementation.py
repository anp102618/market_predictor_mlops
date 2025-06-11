import re
import nltk
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from Common_Utils import setup_logger, track_performance, CustomException

# Setup logger
logger = setup_logger(filename="logs")

# Download required resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


class TextCleaner:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    @staticmethod
    def clean_text(text):
        try:
            text_without_links = re.sub(r'http[s]?://\S+', '', str(text))
            cleaned_text = re.sub(r'[^a-z0-9\-\+\%\s]', '', text_without_links.lower())
            return cleaned_text
        except CustomException as e:
            logger.warning(f"[clean_text] Failed for input: {text} | Error: {e}")
            

    @staticmethod
    def tokenize(text):
        try:
            return word_tokenize(text)
        except CustomException as e:
            logger.warning(f"[tokenize] Tokenization failed: {text} | Error: {e}")

    @classmethod
    def remove_stopwords(cls, tokens):
        try:
            return [word for word in tokens if word not in cls.stop_words]
        except CustomException as e:
            logger.warning(f"[remove_stopwords] Error: {e}")

    @classmethod
    def lemmatize(cls, tokens):
        try:
            return [cls.lemmatizer.lemmatize(word) for word in tokens]
        except CustomException as e:
            logger.warning(f"[lemmatize] Error: {e}")

    @staticmethod
    def remove_noise(text):
        try:
            text = text.lower()
            if "reuters -" in text:
                return text.split("reuters -", 1)[1].strip()
            elif "investingcom --" in text:
                return text.split("investingcom --", 1)[1].strip()
            elif "investingcom" in text:
                return text.split("investingcom", 1)[1].strip()
            elif "investing.com--" in text:
                return text.split("investing.com--", 1)[1].strip()
            elif "investing.com" in text:
                return text.split("investing.com", 1)[1].strip()
            elif "stocktwits -" in text:
                return text.split("stocktwits -", 1)[1].strip()
            elif "ians" in text:
                return text.split("ians", 1)[1].strip()
            else:
                return text
        except CustomException as e:
            logger.warning(f"[remove_noise] Failed for text: {text} | Error: {e}")

    @staticmethod
    def process(df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("[TextCleaner] Cleaning process started")
            df = df.copy()

            df['cleaned_text'] = df['text'].apply(TextCleaner.clean_text)
            df['tokens'] = df['cleaned_text'].apply(TextCleaner.tokenize)
            df['tokens'] = df['tokens'].apply(TextCleaner.remove_stopwords)
            df['tokens'] = df['tokens'].apply(TextCleaner.lemmatize)
            df['joined_text'] = df['tokens'].apply(lambda x: ' '.join(x))
            df['joined_text'] = df['joined_text'].apply(TextCleaner.remove_noise)

            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.sort_values(by='date')

            logger.info(f"[TextCleaner] Completed processing {len(df)} records")
            return df

        except CustomException as e:
            logger.critical(f"[TextCleaner.run] Critical failure during run: {e}", exc_info=True)
            


class SentimentScorer:

    def __init__(self, model_name: str = "yiyanghkust/finbert-tone"):
        try:
            logger.info("[SentimentScorer] Loading model and tokenizer...")
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_name)
            logger.info("[SentimentScorer] Model and tokenizer loaded.")
        except CustomException as e:
            logger.critical(f"[SentimentScorer] Error loading model/tokenizer: {e}", exc_info=True)
            raise

    def get_sentiment_score(self, text: str) -> float:
        """Returns the neutral sentiment score (class index 1) for a single string."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = softmax(logits, dim=-1)
            return probs[0][1].item()  # Index 1 = neutral
        except CustomException as e:
            logger.warning(f"[SentimentScorer] Failed to get sentiment for text: {text[:50]}... | Error: {e}")
            return np.nan

    def add_sentiment_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects a DataFrame with 'date' and 'joined_text' columns.
        Groups by date, calculates sentiment for each article, and adds mean score per date.
        """
        try:
            logger.info("[SentimentScorer] Starting sentiment scoring...")

            df_copy = df.copy()
            df_grouped = df_copy.groupby('date')['joined_text'].apply(list).reset_index()
            df_grouped.columns = ['date', 'news_list']

            df_grouped['sentiment_scores'] = df_grouped['news_list'].apply(
                lambda news_list: [self.get_sentiment_score(str(news)) for news in news_list]
            )

            df_grouped['mean_sentiment_score'] = df_grouped['sentiment_scores'].apply(
                lambda scores: np.nanmean(scores) if scores else np.nan
            )

            df_grouped['date'] = pd.to_datetime(df_grouped['date'], errors='coerce').dt.date
            df_grouped = df_grouped.dropna(subset=['date'])
            df_grouped['date'] = pd.to_datetime(df_grouped['date']).dt.strftime('%Y-%m-%d')

            logger.info(f"[SentimentScorer] Completed sentiment scoring for {len(df_grouped)} days.")
            return df_grouped

        except CustomException as e:
            logger.critical(f"[SentimentScorer] Error during sentiment scoring: {e}", exc_info=True)
            
@track_performance    
def finbert_implement(df):
    try:
        logger.info(f"Text Cleaning starting on news df")
        text_cleaner= TextCleaner()
        df_joined = text_cleaner.process(df)
        logger.info(f"Text Cleaning completed successfully on news df")
        logger.info(f"finbert implementation starting on news df")
        sentiment_scorer = SentimentScorer()
        df_final = sentiment_scorer.add_sentiment_scores(df_joined)
        logger.info(f"finbert implementation completed sucessfully on news df")
        return df_final

    except CustomException as e :
        logger.error(f"Error in Yext cleaning/finbert implementation on df :{e}")
