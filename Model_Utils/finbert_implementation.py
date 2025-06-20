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
from Common_Utils import setup_logger, track_performance, CustomException
from typing import List, Optional

# Setup logger
logger = setup_logger(filename="logs")

# Download required resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


class TextCleaner:
    """
    Utility class for text cleaning operations like lowercasing, noise removal,
    tokenization, stopword removal, and lemmatization.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove URLs and unwanted characters from text."""
        try:
            text_without_links = re.sub(r'http[s]?://\S+', '', str(text))
            cleaned_text = re.sub(r'[^a-z0-9\-\+\%\s]', '', text_without_links.lower())
            return cleaned_text
        except CustomException as e:
            logger.warning(f"[clean_text] Failed: {text} | Error: {e}")
            return ""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text using NLTK word_tokenize."""
        try:
            return word_tokenize(text)
        except CustomException as e:
            logger.warning(f"[tokenize] Failed: {text} | Error: {e}")
            return []

    @classmethod
    def remove_stopwords(cls, tokens: List[str]) -> List[str]:
        """Remove stopwords from a list of tokens."""
        try:
            return [word for word in tokens if word not in cls.stop_words]
        except CustomException as e:
            logger.warning(f"[remove_stopwords] Error: {e}")
            return tokens

    @classmethod
    def lemmatize(cls, tokens: List[str]) -> List[str]:
        """Lemmatize a list of tokens."""
        try:
            return [cls.lemmatizer.lemmatize(word) for word in tokens]
        except CustomException as e:
            logger.warning(f"[lemmatize] Error: {e}")
            return tokens

    @staticmethod
    def remove_noise(text: str) -> str:
        """Remove known noise patterns from news headlines or body."""
        try:
            text = text.lower()
            noise_sources = [
                "reuters -", "investingcom --", "investingcom", "investing.com--",
                "investing.com", "stocktwits -", "ians"
            ]
            for noise in noise_sources:
                if noise in text:
                    return text.split(noise, 1)[-1].strip()
            return text
        except CustomException as e:
            logger.warning(f"[remove_noise] Failed: {text} | Error: {e}")
            return text

    @staticmethod
    def process(df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning pipeline: clean text, tokenize, remove stopwords,
        lemmatize, join, remove noise, and sort by date.
        """
        try:
            logger.info("[TextCleaner] Starting processing")
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

            logger.info(f"[TextCleaner] Completed cleaning {len(df)} records")
            return df

        except CustomException as e:
            logger.critical(f"[TextCleaner.process] Critical failure: {e}")
            


class SentimentScorer:
    """
    Sentiment analysis wrapper for FinBERT model, extracting neutral sentiment scores.
    """

    def __init__(self, model_name: str = "yiyanghkust/finbert-tone"):
        try:
            logger.info("[SentimentScorer] Loading FinBERT...")
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_name)
            logger.info("[SentimentScorer] FinBERT loaded.")
        except CustomException as e:
            logger.critical(f"[SentimentScorer] Model loading failed: {e}")
            
    def get_sentiment_score(self, text: str) -> float:
        """Compute neutral sentiment score (class index 1) for a given text."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = softmax(logits, dim=-1)
            return probs[0][1].item()  # Neutral class
        
        except CustomException as e:
            logger.warning(f"[get_sentiment_score] Failed: {text[:50]}... | Error: {e}")
            return np.nan

    def add_sentiment_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group news by date, compute sentiment for each, and return mean score per day.
        """
        try:
            logger.info("[SentimentScorer] Starting sentiment analysis...")

            df_copy = df.copy()
            df_grouped = df_copy.groupby('date')['joined_text'].apply(list).reset_index()
            df_grouped.columns = ['date', 'news_list']

            df_grouped['sentiment_scores'] = df_grouped['news_list'].apply(
                lambda texts: [self.get_sentiment_score(str(txt)) for txt in texts]
            )

            df_grouped['mean_sentiment_score'] = df_grouped['sentiment_scores'].apply(
                lambda scores: np.nanmean(scores) if scores else np.nan
            )

            df_grouped['date'] = pd.to_datetime(df_grouped['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            df_grouped = df_grouped.dropna(subset=['date'])

            logger.info(f"[SentimentScorer] Completed {len(df_grouped)} sentiment entries.")
            return df_grouped

        except CustomException as e:
            logger.critical(f"[add_sentiment_scores] Sentiment processing failed: {e}")
            


@track_performance
def finbert_implement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline to clean news data and apply FinBERT sentiment scoring.

    Args:
        df (pd.DataFrame): Input dataframe with 'text' and 'date' columns.

    Returns:
        pd.DataFrame: Output with daily mean sentiment scores.
    """
    try:
        logger.info("[finbert_implement] Cleaning news text...")
        cleaned_df = TextCleaner.process(df)

        logger.info("[finbert_implement] Running FinBERT sentiment analysis...")
        scorer = SentimentScorer()
        final_df = scorer.add_sentiment_scores(cleaned_df)

        logger.info("[finbert_implement] Completed sentiment pipeline.")
        return final_df

    except CustomException as e:
        logger.error(f"[finbert_implement] Pipeline failed: {e}")
        
