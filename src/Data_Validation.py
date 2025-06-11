import pandas as pd
from datetime import datetime
from Common_Utils import setup_logger, track_performance, CustomException
from DataBase.db_handler import SQLiteEditor

# ------------------ Logger Setup ------------------ #
# Setup logger
logger = setup_logger(filename="logs")
# ------------------ DataValidator Class ------------------ #
class DataValidator:
    def __init__(self):
        db = SQLiteEditor()
        self.df = db.fetch_df(query="SELECT * FROM final_data")
        self.df = self.df.dropna()
        self.errors = []
        self.required_cols = [
            'date', 'mean_sentiment_score', 'nasdaq', 'sp500', 'dj30',
            'crude_oil', 'gold', 'usd_inr', '10yb', 'vix', 'nsebank', 'nsei'
        ]
    @track_performance
    def check_columns(self):
        missing = [col for col in self.required_cols if col not in self.df.columns]
        if missing:
            msg = f"Missing required columns: {missing}"
            self.errors.append(msg)
            logger.error(msg)

    @track_performance
    def check_dtypes(self):
        try:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            invalid_date_rows = self.df[self.df['date'].isnull()]
            if not invalid_date_rows.empty:
                msg = f"Invalid 'date' values at rows: {invalid_date_rows.index.tolist()}"
                self.errors.append(msg)
                logger.error(msg)
        except CustomException as e:
            logger.exception("Failed during date conversion.")

        numeric_cols = self.required_cols[1:]
        for col in numeric_cols:
            try:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                invalid = self.df[self.df[col].isnull()]
                if not invalid.empty:
                    msg = f"Non-numeric or NaN values in '{col}' at rows: {invalid.index.tolist()}"
                    self.errors.append(msg)
                    logger.error(msg)
            except CustomException as e:
                logger.exception(f"Failed to convert column '{col}' to numeric.")

    @track_performance
    def check_missing_values(self):
        try:
            # 1. Drop columns with ≥ 40% missing values
            col_thresh = 0.4
            col_missing_ratio = self.df.isnull().mean()
            cols_to_drop = col_missing_ratio[col_missing_ratio >= col_thresh].index.tolist()
            if cols_to_drop:
                self.df.drop(columns=cols_to_drop, inplace=True)
                msg = f"Dropped columns with ≥ {int(col_thresh * 100)}% missing values: {cols_to_drop}"
                self.errors.append(msg)
                logger.warning(msg)

            # 2. Drop rows with ≥ 40% missing values
            row_thresh = 0.4
            row_missing_ratio = self.df.isnull().mean(axis=1)
            rows_to_drop = self.df.index[row_missing_ratio >= row_thresh].tolist()
            if rows_to_drop:
                self.df.drop(index=rows_to_drop, inplace=True)
                msg = f"Dropped rows with ≥ {int(row_thresh * 100)}% missing values: {rows_to_drop}"
                self.errors.append(msg)
                logger.warning(msg)

            # 3. Log remaining rows with any missing values
            null_rows = self.df[self.df.isnull().any(axis=1)]
            if not null_rows.empty:
                msg = f"Remaining missing values at rows: {null_rows.index.tolist()}"
                self.errors.append(msg)
                logger.error(msg)

        except CustomException as e:
            logger.error(f"Error while checking for missing values: {e}")
        


    @track_performance
    def check_value_ranges(self):
        try:
            invalid_sentiment = self.df[
                (self.df['mean_sentiment_score'] < -1) | 
                (self.df['mean_sentiment_score'] > 1)
            ]
            if not invalid_sentiment.empty:
                msg = f"'mean_sentiment_score' out of range [-1,1] at rows: {invalid_sentiment.index.tolist()}"
                self.errors.append(msg)
                logger.error(msg)
        except CustomException as e:
            logger.exception("Error while checking sentiment score range.")

        non_negative_cols = [
            'nasdaq', 'sp500', 'dj30', 'crude_oil', 'gold',
            'usd_inr', '10yb', 'vix', 'nsebank', 'nsei'
        ]
        for col in non_negative_cols:
            try:
                negatives = self.df[self.df[col] < 0]
                if not negatives.empty:
                    msg = f"Negative values in '{col}' at rows: {negatives.index.tolist()}"
                    self.errors.append(msg)
                    logger.error(msg)
            except CustomException as e:
                logger.exception(f"Error while checking non-negative values for '{col}'.")

    @track_performance
    def check_duplicates(self):
        try:
            duplicates = self.df[self.df.duplicated()]
            if not duplicates.empty:
                msg = f"Duplicate rows at indices: {duplicates.index.tolist()}"
                self.errors.append(msg)
                logger.error(msg)
        except CustomException as e:
            logger.exception("Error while checking for duplicate rows.")
    
    @track_performance
    def check_future_dates(self):
        try:
            future = self.df[self.df['date'] > pd.to_datetime('today')]
            if not future.empty:
                msg = f"Future dates found at rows: {future.index.tolist()}"
                self.errors.append(msg)
                logger.error(msg)
        except CustomException as e:
            logger.exception("Error while checking future dates.")

    @track_performance
    def run_all_checks(self):
        try:
            logger.info("Starting data validation checks...")
            self.check_columns()
            self.check_dtypes()
            self.check_missing_values()
            self.check_value_ranges()
            self.check_duplicates()
            self.check_future_dates()
            is_valid = len(self.errors) == 0
            if not is_valid:
                logger.warning("Data validation failed with the following issues:")
                for err in self.errors:
                    logger.warning(err)

            if self.errors:
                logger.warning("Data validation failed.")
                raise ValueError("Data validation failed. Check the logs for details.")
            else:
                logger.info(f"Data validation successful. Rows checked: {len(self.df)}")
                print(f"Data validation successful. Rows checked: {len(self.df)}")
            
            return is_valid, self.errors

        except CustomException as e:
            logger.error(f"Data Validation failed: {e}")

def execute_data_validation():

    try:
        logger.info("Starting Data Validation pipeline")
        data_validator = DataValidator()
        print(data_validator.run_all_checks())
        logger.info(" Data Validation pipeline sucessfully executed")
    
    except CustomException as e:
            logger.error(f"Data Validation process failed: {e}")


if __name__ == "__main__":
    try:
        logger.info("Starting Data Validation pipeline")

        execute_data_validation()

        logger.info(" Data Validation pipeline sucessfully executed")
        
    
    except CustomException as e:
            logger.error(f"Data Validation process failed: {e}")