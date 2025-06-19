# news_scraper.py
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, WebDriverException
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import time
from abc import ABC, abstractmethod
from Common_Utils import setup_logger, track_performance, CustomException, load_yaml
from Model_Utils.finbert_implementation import finbert_implement
from DataBase.db_handler import SQLiteEditor

# Setup logger
logger = setup_logger(filename="logs")
config = load_yaml("Config_Yaml/model_config.yaml")

#--------------- Config-path---------------#
db_path: Path = Path(config["Data_Ingestion"]["path"]["database"])          # "Data/data.db"
news_data_table: str = config["Data_Ingestion"]["path"]["news_data_table"]    # "news_data"
ticker_data_table: str = config["Data_Ingestion"]["path"]["ticker_data_table"]  # "ticker_data"
final_data_table: str = config["Data_Ingestion"]["path"]["final_data_table"]   # "final_data"
raw_finbert_csv: Path = Path(config["Data_Ingestion"]["path"]["raw_finbert_csv"])    # "Data/raw_data/finbert_data.csv"
new_news_csv: Path = Path(config["Data_Ingestion"]["path"]["new_news_data"])      # "Data/new_data/news_data.csv"
raw_news_csv: Path = Path(config["Data_Ingestion"]["path"]["raw_news_data"])      # "Data/raw_data/news_data.csv"
new_ticker_csv: Path = Path(config["Data_Ingestion"]["path"]["new_ticker_csv"])     # "Data/new_data/ticker_data.csv"
raw_ticker_csv: Path = Path(config["Data_Ingestion"]["path"]["raw_ticker_csv"])     # "Data/raw_data/ticker_data.csv"
final_data_csv: Path = Path(config["Data_Ingestion"]["path"]["final_data_csv"])     # "Data/processed_data/final_data.csv"
final_data_new:Path = Path(config["Data_Ingestion"]["path"]["final_data_new"])      # "Data/new_data/final_data.csv"

#-------------------Config-const------------------#
base_url: str = config["Data_Ingestion"]["const"]["base_url"]      # "https://in.investing.com/indices/s-p-cnx-nifty-news/"
links_xpath: str = config["Data_Ingestion"]["const"]["links_xpath"]   # "//ul[@data-test='news-list']//article[@data-test='article-item']//div[@class='relative']//a"
article_xpath: str = config["Data_Ingestion"]["const"]["article_xpath"] # "//*[@id='article'][@class='article_container']//p"
date_xpath: str = config["Data_Ingestion"]["const"]["date_xpath"]    # "//span[contains(text(),'Published')]"
table_xpath: str = config["Data_Ingestion"]["const"]["table_xpath"]   # "//div[@class='container' and @data-testid='history-table']//table"



class WebDataScraper(ABC):
    @abstractmethod
    def update_data(self):
        pass

class NewsDataUpdater(WebDataScraper):
    def __init__(self, db_path= db_path, table_name= news_data_table):
        self.base_url = base_url
        self.links_xpath = links_xpath
        self.db = SQLiteEditor(db_path)
        self.table_name = table_name
        self.chrome_options = Options()
        self.chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                                         " AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")
        self.chrome_options.add_argument('--headless')
        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.managed_default_content_settings.stylesheets": 2,
            "profile.managed_default_content_settings.cookies": 2,
            "profile.managed_default_content_settings.javascript": 1,  # Keep JS enabled if needed
            "profile.managed_default_content_settings.plugins": 2,
            "profile.managed_default_content_settings.popups": 2,
            "profile.managed_default_content_settings.geolocation": 2,
            "profile.managed_default_content_settings.media_stream": 2,
            "profile.managed_default_content_settings.fonts": 2,
            "profile.managed_default_content_settings.notifications": 2,
        }
        self.chrome_options.add_experimental_option("prefs", prefs)
        self.chrome_options.add_argument("--ignore-certificate-errors")
        self.chrome_options.add_argument("--incognito")
        self.chrome_options.add_argument("--disable-extensions")
        

    @track_performance
    def get_last_date_from_db(self):
        try:
            query = f"SELECT * FROM {self.table_name}"
            df = self.db.fetch_df(query)
            if not df.empty and "date" in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                last_date = df['date'].max().date()
                logger.info(f"[DB] Last news date: {last_date}")
                return last_date
            else:
                logger.warning("[DB] No records found. Defaulting to 7 days ago.")
                return datetime.now().date() - timedelta(days=7)
        except CustomException as e:
            logger.error(f"[DB ERROR] Failed to fetch last date: {e}")
            return datetime.now().date() - timedelta(days=7)

    @track_performance
    def extract_text(self, url, max_retries=3, wait_between_retries=3):
        for attempt in range(1, max_retries + 1):
            driver = None
            try:
                logger.info(f"[SCRAPER] Attempt {attempt} for URL: {url}")
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.chrome_options)
                driver.get(url)

                # Wait for article paragraphs
                paragraphs = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located(
                        (By.XPATH, article_xpath)
                    )
                )

                # Wait for the published date
                date_text = WebDriverWait(driver, 10).until(
                    EC.visibility_of_element_located(
                        (By.XPATH, date_xpath)
                    )
                ).text

                # Convert the extracted date
                extracted_date = datetime.strptime(
                    date_text.split(" ")[1].strip().replace(",", ""),
                    "%d-%m-%Y"
                ).strftime("%Y-%m-%d")

                # Join all paragraph texts
                full_text = " ".join([p.text for p in paragraphs if p.text.strip()])
                
                logger.info(f"[SCRAPER] Successfully extracted article for URL: {url}")
                return [extracted_date, full_text]

            except (TimeoutException, WebDriverException, Exception) as e:
                logger.warning(f"[SCRAPE RETRY {attempt}] URL: {url} - Error: {str(e)}")
                if attempt < max_retries:
                    time.sleep(wait_between_retries)
                else:
                    logger.error(f"[SCRAPE FAILED] Max retries reached for URL: {url}")
                    return None

            finally:
                if driver:
                    driver.quit()

    @track_performance
    def update_data(self):
        try:
            logger.info(f"Starting update_dataframe for news_data table ")
            last_date = self.get_last_date_from_db()
            end_date = datetime.now().date() - timedelta(days=1)
            all_data = pd.DataFrame(columns=["date", "text"])
            page = 1
            max_retries = 3

            while True:
                try:
                    retry_count = 0
                    success = False

                    while retry_count < max_retries and not success:
                        try:
                            logger.info(f"[SCRAPE] Fetching page {page}")
                            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.chrome_options)
                            driver.get(f"{self.base_url}{page}")
                            elements = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, self.links_xpath)))
                            links = [el.get_attribute("href") for el in elements]
                            driver.quit()

                            with ThreadPoolExecutor(max_workers=1) as executor:
                                results = list(executor.map(self.extract_text, links))

                            filtered = [r for r in results if r]
                            if not filtered:
                                logger.info("[SCRAPE] No new articles found. Ending loop.")
                                break

                            df = pd.DataFrame(filtered, columns=["date", "text"])
                            df["date"] = pd.to_datetime(df["date"]).dt.date

                            if (df["date"] <= last_date).any():
                                df = df[(df["date"] > last_date) & (df["date"] <= end_date)]
                                if not df.empty:
                                    all_data = pd.concat([all_data, df], ignore_index=True)
                                success = True
                                logger.info("[SCRAPE] Reached articles already present. Ending.")
                                break
                            else:
                                df = df[df["date"] <= end_date]
                                if not df.empty:
                                    all_data = pd.concat([all_data, df], ignore_index=True)
                                page += 1
                                time.sleep(2)

                        except CustomException as e:
                            retry_count += 1
                            logger.error(f"[RETRY] Page {page} attempt {retry_count} failed: {e}")
                            time.sleep(3)
                    
                    if success:
                        break
            
                except CustomException as e :
                    logger.error(f"Error occured in execution:{e}")


                finally:
                    if driver:
                        driver.quit()

            df_finbert = finbert_implement(all_data)
            df_finbert.to_csv(new_news_csv)
            df1 = pd.read_csv(raw_finbert_csv)
            df1 = df1[["date", "news_list", "sentiment_scores", "mean_sentiment_score"]]
            df2 = df_finbert[["date", "news_list", "sentiment_scores", "mean_sentiment_score"]]
            finbert_data = pd.concat([df1, df2], ignore_index=True)
            finbert_data['date'] = pd.to_datetime(finbert_data['date'], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')
            finbert_data.to_csv(raw_finbert_csv)
            logger.info(f"Sucessfully completed  update  new_data/news_data.csv ")
            
            df_finbert = df_finbert[["date","mean_sentiment_score"]]
            query = f"SELECT * FROM {self.table_name}"
            df_existing = self.db.fetch_df(query)
            df_existing['date'] = pd.to_datetime(df_existing['date']).dt.strftime('%Y-%m-%d')
            if 'Unnamed: 0' in df_existing.columns:
                df_existing = df_existing.drop(columns=['Unnamed: 0'])

            df_final = pd.concat([df_existing, df_finbert], ignore_index=True)
            df_final['date'] = pd.to_datetime(df_final['date']).dt.strftime('%Y-%m-%d')
            print(df_final.tail(2))

            df_final.to_csv(raw_news_csv)
            logger.info(f"Sucessfully completed  update_dataframe for news_data.csv ")
            
            self.db.write_df(df=df_final[["date","mean_sentiment_score"]], csv_path=raw_news_csv, table_name=self.table_name)
            logger.info(f"Sucessfully completed  update_dataframe for news_data table ")

        except CustomException as e :
                    logger.error(f"Error occured in execution:{e}")
                    

class TickerDataUpdater(WebDataScraper):
    def __init__(self, db_path=db_path, table_name= ticker_data_table):
        self.db_path = db_path
        self.table_name = table_name
        self.db = SQLiteEditor(db_path)

        self.col_to_ticker = {
            'nasdaq': '%5EIXIC',
            'dj30': '%5EDJI',
            'sp500': '%5EGSPC',
            'gold': 'GC%3DF',
            'crude_oil': 'CL%3DF',
            'usd_inr': 'INR%3DX',
            'nsebank': '%5ENSEBANK',
            '10yb': '%5ETNX',
            'vix': '%5EVIX',
            'nsei': '%5ENSEI'
        }

        self.chrome_options = Options()
        self.chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument("--ignore-certificate-errors")
        self.chrome_options.add_argument("--allow-running-insecure-content")
        self.chrome_options.add_argument("--disable-popup-blocking")
        self.chrome_options.add_argument("--incognito") 
        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.managed_default_content_settings.stylesheets": 2,
            "profile.managed_default_content_settings.cookies": 2,
            "profile.managed_default_content_settings.javascript": 1,  # Keep JS enabled if needed
            "profile.managed_default_content_settings.plugins": 2,
            "profile.managed_default_content_settings.popups": 2,
            "profile.managed_default_content_settings.geolocation": 2,
            "profile.managed_default_content_settings.media_stream": 2,
            "profile.managed_default_content_settings.fonts": 2,
            "profile.managed_default_content_settings.notifications": 2,
        }
        self.chrome_options.add_experimental_option("prefs", prefs)


    @track_performance
    def get_closing_prices(self, ticker):
        max_retries = 3
        url = f'https://finance.yahoo.com/quote/'+ticker+'/history'

        for attempt in range(max_retries):
            driver = None
            try:
                logger.info(f"[{ticker}] Attempt {attempt+1}: Fetching {url}")
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.chrome_options)
                driver.get(url)

                table = driver.find_element(By.XPATH, table_xpath)
                rows = table.find_elements(By.TAG_NAME, 'tr')

                data = []
                for row in rows:
                    cols = row.find_elements(By.TAG_NAME, 'td')
                    if not cols:
                        cols = row.find_elements(By.TAG_NAME, 'th')
                    data.append([col.text.strip() for col in cols])

                df = pd.DataFrame(data)
                df.columns = df.iloc[0]
                df = df[1:]
                df.rename(columns={'Date': 'date', 'Close': 'close'}, inplace=True)

                # Parse dates and close prices
                df['date'] = pd.to_datetime(df['date'].apply(lambda x: datetime.strptime(x, "%b %d, %Y")))
                df['close'] = df['close'].apply(lambda x: float(x.replace(",", "")))

                logger.info(f"[{ticker}] Successfully scraped {len(df)} rows.")
                return df[['date', 'close']]

            except TimeoutException as e:
                logger.warning(f"[{ticker}] Attempt {attempt+1} timeout error: {e}")
            except WebDriverException as e:
                logger.warning(f"[{ticker}] Attempt {attempt+1} WebDriver error: {e}")
            except Exception as e:
                logger.error(f"[{ticker}] Attempt {attempt+1} failed with error: {e}", exc_info=True)
            finally:
                if driver:
                    driver.quit()

            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                logger.critical(f"[{ticker}] All {max_retries} attempts failed to scrape data from {url}")
            

    @track_performance
    def get_last_date_from_db(self):
        try:
            query = f"SELECT * FROM {self.table_name}"
            df = self.db.fetch_df(query)
            if not df.empty and "date" in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                last_date = df['date'].max().date()
                logger.info(f"[DB] Last news date: {last_date}")
                return last_date
            else:
                logger.warning("[DB] No records found. Defaulting to 7 days ago.")
                return datetime.now().date() - timedelta(days=7)
        except CustomException as e:
            logger.error(f"[DB ERROR] Failed to fetch last date: {e}")
            return datetime.now().date() - timedelta(days=7)
            
    @track_performance
    def update_data(self):
        try:
            query = f"SELECT * FROM {self.table_name}"
            df_existing = self.db.fetch_df(query)
            df_existing['date'] = pd.to_datetime(df_existing['date']).dt.strftime('%Y-%m-%d')
            df_existing = df_existing.iloc[:, 1:]
            if 'Unnamed: 0' in df_existing.columns:
                df_existing = df_existing.drop(columns=['Unnamed: 0'])
            logger.info(f"Existing ticker_df columns :{df_existing.columns}")
            last_date = self.get_last_date_from_db()
            yesterday = datetime.now().date() - timedelta(days=1)
            target_dates = pd.date_range(start=last_date + timedelta(days=1), end=yesterday)

            if len(target_dates) == 0:
                logger.info(" No new dates to fetch. Table is already up to date.")
                return
            df_new = pd.DataFrame({'date': pd.to_datetime(target_dates)})

            for col, ticker in self.col_to_ticker.items():
                time.sleep(5)
                logger.info(f"Start get_closing_price for ticker :{ticker}")
                df_ticker = self.get_closing_prices(ticker)
                if df_ticker is None or df_ticker.empty or 'date' not in df_ticker or 'close' not in df_ticker:
                    df_filtered = pd.DataFrame({'date': target_dates, col: np.nan})
                else:
                    df_ticker['date'] = pd.to_datetime(df_ticker['date'])
                    df_filtered = df_ticker[df_ticker['date'].isin(target_dates)].rename(columns={'close': col})
                    df_filtered = pd.merge(pd.DataFrame({'date': pd.to_datetime(target_dates)}), df_filtered, on='date', how='left')
                    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
                    df_new['date'] = pd.to_datetime(df_new['date'])
                    df_new = df_new.merge(df_filtered, on='date', how='left')
                    df_new['date'] = pd.to_datetime(df_new['date']).dt.date
                    df_new['date'] = pd.to_datetime(df_new['date']).dt.strftime('%Y-%m-%d')
                    logger.info(f" get_closing_price successfully executed for ticker :{ticker}")

            df_new.drop_duplicates(subset='date', keep='last', inplace=True)
            df_new.to_csv(new_ticker_csv)
            logger.info("Successfully updated  new_ticker_data.csv with new stock data.")
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.dropna(subset=['nsei'], inplace=True)
            df_combined.bfill(inplace=True)
            df_combined.drop_duplicates(subset='date', keep='last', inplace=True)
            df_combined = df_combined.reset_index(drop=True)
            df_combined['date'] = pd.to_datetime(df_combined['date']).dt.strftime('%Y-%m-%d')
            df_combined = df_combined.round(2)
            print(df_combined.tail(2))
            df_combined.to_csv(raw_ticker_csv)
            logger.info("Successfully updated ticker_data.csv with new stock data.")
            self.db.write_df(df=df_combined, csv_path=raw_ticker_csv, table_name=self.table_name)
            logger.info("Successfully updated db table ticker_data with new stock data.")
            

        except CustomException as e:
            logger.error(f" DataFrame update failed: {e}")

class ScraperContext:
    def __init__(self, scraper: WebDataScraper):
        self.scraper = scraper

    def set_scraper(self, scraper: WebDataScraper):
        self.scraper = scraper

    def run(self):
        return self.scraper.update_data()

@track_performance
def align_sentiment_with_price(table_name1:str = ticker_data_table, table_name2:str = news_data_table ):
    """
    Aligns sentiment scores (df2) with stock prices (df1) based on date,
    filling missing price segments with average sentiment scores.
    """
    try:
        logger.info("Reading tables in db as df....")
        conn = sqlite3.connect("Data/data.db")  # Example: "data/my_data.db"
        df1 = pd.read_sql_query(f"SELECT * FROM {table_name1}", conn)
        df2 = pd.read_sql_query(f"SELECT * FROM {table_name2}", conn)
        conn.close()

        logger.info("Merging sentiment data with price data...")
        merged = pd.merge(df2, df1, on='date', how='left')
        merged = merged.sort_values('date').reset_index(drop=True)

        scores = merged['mean_sentiment_score'].tolist()
        closes = merged['nsei'].tolist()
        result_scores = scores.copy()

        start_idx = None
        for i, val in enumerate(closes):
            if pd.isna(val):
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None:
                    segment = list(range(start_idx, i + 1))
                    avg_score = np.nanmean([scores[j] for j in segment])
                    for j in segment:
                        result_scores[j] = avg_score
                    start_idx = None

        # Handle trailing NaNs in price
        if start_idx is not None:
            segment = list(range(start_idx, len(scores)))
            avg_score = np.nanmean([scores[j] for j in segment])
            for j in segment:
                result_scores[j] = avg_score

        # Update the merged DataFrame
        merged['mean_sentiment_score'] = result_scores

        # Filter to keep only dates present in df1
        final_df = merged[merged['date'].isin(df1['date'])].reset_index(drop=True)
        logger.info("Sentiment alignment completed successfully.")

        final_df.to_csv(final_data_csv)
        final_df.to_csv(final_data_new)
        logger.info("Successfully updated ticker_data.csv with new stock data.")
        

    except CustomException as e:
        logger.critical(f"Failed during sentiment-price alignment :{e}", exc_info=True)
        

@track_performance
def execute_data_ingestion():
    try:
        logger.info("Starting Data Ingestion pipeline")
        db = SQLiteEditor()
        db.initiate_dbs()
        data_scraper = ScraperContext(TickerDataUpdater())
        data_scraper.run()
        data_scraper.set_scraper(NewsDataUpdater())
        data_scraper.run()
        align_sentiment_with_price()
        df_final = pd.read_csv(final_data_csv, index_col=[0])
        db.write_df(df=df_final, csv_path=final_data_csv, table_name=final_data_table)
        logger.info(" Data Ingestion pipeline sucessfully executed")
        
    
    except CustomException as e:
            logger.error(f"[Web Data Updater]  update failed: {e}")



if __name__ == "__main__":
    try:
        logger.info("Starting Data Ingestion pipeline")

        execute_data_ingestion()

        logger.info(" Data Ingestion pipeline sucessfully executed")
        
    
    except CustomException as e:
            logger.error(f"Data Ingestion process failed: {e}")