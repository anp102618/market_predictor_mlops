import pandas as pd
from datetime import datetime
from Common_Utils import setup_logger, track_performance, CustomException
from DataBase.db_handler import SQLiteEditor
from typing import Tuple, List

# ------------------ Logger Setup ------------------ #
logger = setup_logger(filename="logs")

class DataValidator:
    """
    Validates a DataFrame from the final_data table in SQLite.
    Checks include column presence, data types, missing values, value ranges, duplicates, and future dates.
    """
    def __init__(self):
        self.db = SQLiteEditor()
        self.df = self.db.fetch_df(query="SELECT * FROM final_data").dropna()
        self.errors: List[str] = []
        self.required_cols = [
            'date', 'mean_sentiment_score', 'nasdaq', 'sp500', 'dj30',
            'crude_oil', 'gold', 'usd_inr', '10yb', 'vix', 'nsebank', 'nsei'
        ]

    @track_performance
    def check_columns(self) -> None:
        """Checks if all required columns are present in the DataFrame."""
        missing = [col for col in self.required_cols if col not in self.df.columns]
        if missing:
            msg = f"Missing required columns: {missing}"
            self.errors.append(msg)
            logger.error(msg)

    @track_performance
    def check_dtypes(self) -> None:
        """Checks that 'date' is a valid datetime and numeric columns contain valid numbers."""
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
    def check_missing_values(self) -> None:
        """Drops rows/columns with excessive missing values and logs residual nulls."""
        try:
            col_thresh = 0.4
            row_thresh = 0.4

            col_missing_ratio = self.df.isnull().mean()
            cols_to_drop = col_missing_ratio[col_missing_ratio >= col_thresh].index.tolist()
            if cols_to_drop:
                self.df.drop(columns=cols_to_drop, inplace=True)
                msg = f"Dropped columns with ≥ {int(col_thresh * 100)}% missing values: {cols_to_drop}"
                self.errors.append(msg)
                logger.warning(msg)

            row_missing_ratio = self.df.isnull().mean(axis=1)
            rows_to_drop = self.df.index[row_missing_ratio >= row_thresh].tolist()
            if rows_to_drop:
                self.df.drop(index=rows_to_drop, inplace=True)
                msg = f"Dropped rows with ≥ {int(row_thresh * 100)}% missing values: {rows_to_drop}"
                self.errors.append(msg)
                logger.warning(msg)

            null_rows = self.df[self.df.isnull().any(axis=1)]
            if not null_rows.empty:
                msg = f"Remaining missing values at rows: {null_rows.index.tolist()}"
                self.errors.append(msg)
                logger.error(msg)

        except CustomException as e:
            logger.exception("Error while checking for missing values.")

    @track_performance
    def check_value_ranges(self) -> None:
        """Checks that sentiment scores are within [-1, 1] and financial indicators are non-negative."""
        try:
            invalid_sentiment = self.df[(self.df['mean_sentiment_score'] < -1) | (self.df['mean_sentiment_score'] > 1)]
            if not invalid_sentiment.empty:
                msg = f"'mean_sentiment_score' out of range [-1,1] at rows: {invalid_sentiment.index.tolist()}"
                self.errors.append(msg)
                logger.error(msg)
        except Exception as e:
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
    def check_duplicates(self) -> None:
        """Checks for and logs duplicate rows."""
        try:
            duplicates = self.df[self.df.duplicated()]
            if not duplicates.empty:
                msg = f"Duplicate rows at indices: {duplicates.index.tolist()}"
                self.errors.append(msg)
                logger.error(msg)
        except CustomException as e:
            logger.exception("Error while checking for duplicate rows.")

    @track_performance
    def check_future_dates(self) -> None:
        """Checks for date entries that are in the future."""
        try:
            future = self.df[self.df['date'] > pd.to_datetime('today')]
            if not future.empty:
                msg = f"Future dates found at rows: {future.index.tolist()}"
                self.errors.append(msg)
                logger.error(msg)
        except CustomException as e:
            logger.exception("Error while checking future dates.")

    @track_performance
    def run_all_checks(self) -> Tuple[bool, List[str]]:
        """
        Runs all validation checks.

        Returns:
            Tuple[bool, List[str]]: Whether validation passed, and list of errors if any.
        """
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
                for err in self.errors:
                    logger.warning(err)
                raise ValueError("Data validation failed. Check the logs for details.")
            else:
                logger.info(f"Data validation successful. Rows checked: {len(self.df)}")
                print(f"Data validation successful. Rows checked: {len(self.df)}")

            return is_valid, self.errors

        except CustomException as e:
            logger.exception("Data Validation failed")
            return False, self.errors

def execute_data_validation() -> None:
    """Executes the full data validation pipeline."""
    try:
        logger.info("Starting Data Validation pipeline")
        data_validator = DataValidator()
        print(data_validator.run_all_checks())
        logger.info("Data Validation pipeline successfully executed")

    except CustomException as e:
        logger.exception("Data Validation process failed")

if __name__ == "__main__":
    try:
        logger.info("Starting Data Validation pipeline")
        execute_data_validation()
        logger.info("Data Validation pipeline successfully executed")

    except CustomException as e:
        logger.exception("Data Validation process failed")
