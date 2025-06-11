import pandas as pd
import sqlite3 
import ast
from Common_Utils import setup_logger, track_performance, CustomException

# Setup logger
logger = setup_logger(filename="logs")

class SQLiteEditor:
    def __init__(self, db_path="Data/data.db"):
        self.db_path = db_path

    @track_performance
    def connect(self):
        try:
            return sqlite3.connect(self.db_path)
        except CustomException as e:
            logger.error(f"[DB CONNECT ERROR] {e}")
            return None

    @track_performance
    def fetch_df(self, query):
        try:
            conn = self.connect()
            if conn:
                df = pd.read_sql(query, conn)
                conn.close()
                for col in ["news_list", "sentiment_scores"]:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                return df
            else:
                return pd.DataFrame()
        except CustomException as e:
            logger.error(f"[FETCH ERROR] {e}")
            

    @track_performance
    def write_df(self, df: pd.DataFrame, csv_path: str, table_name: str):
        try:
            
            conn = self.connect()
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                    df[col] = df[col].apply(str)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            logger.info(f"[DB WRITE] Wrote {len(df)} records to table '{table_name}' from file '{csv_path}'")
        except Exception as e: 
            logger.error(f"[DB WRITE ERROR] {e}")
            raise
        
    @track_performance
    def initiate_dbs(self, ticker_data_path:str ="Data/raw_data/ticker_data.csv", news_data_path:str ="Data/raw_data/news_data.csv"):

        try:
            logger.info(f"starting dbs table formation ")
            ticker_data =pd.read_csv(ticker_data_path)
            news_data =pd.read_csv(news_data_path)
            news_data = news_data[["date", "mean_sentiment_score"]]
            self.write_df(df=ticker_data, csv_path=ticker_data_path, table_name="ticker_data")
            self.write_df(df=news_data, csv_path=news_data_path, table_name="news_data")
            logger.info(f"dbs table formation completed successfully")
        except Exception as e: 
            logger.error(f"[DB WRITE ERROR] {e}")
            raise

