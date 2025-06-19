# common_utils/__init__.py
import os
import sys
from datetime import datetime
import time
import tracemalloc
from functools import wraps
import yaml
from pathlib import Path
import shutil
import subprocess
import logging
from typing import List

# ─────────────── Global Paths ───────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Cache for loggers by filename
_logger_registry = {}


def setup_logger(name="Common_Utils", filename=None):
    """
    Sets up and returns a logger with optional filename (timestamped).

    Args:
        name (str): Logger name.
        filename (str): Base log filename (e.g. 'text_generation' — timestamp will be added).

    Returns:
        logging.Logger: Configured logger.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Default: use logger name as base for filename
    if filename is None:
        filename = f"{name}_{timestamp}.log"
    else:
        base = Path(filename).stem
        filename = f"{base}_{timestamp}.log"

    if filename in _logger_registry:
        return _logger_registry[filename]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        log_path = LOG_DIR / filename

        # File Handler
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(filename)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    _logger_registry[filename] = logger
    return logger



# ─── Error Utilities ─────────────────────────────
def error_message_detail(error, error_detail: sys):
    """Extracts detailed error information including file name and line number."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    return f"Error in script: [{file_name}] at line [{exc_tb.tb_lineno}] - Message: [{str(error)}]"


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


# ─── Global Exception Hook ───────────────────────
def global_exception_handler(exc_type, exc_value, exc_traceback):
    logger = setup_logger()
    logger.critical("Unhandled Exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = global_exception_handler


# ─── Decorators & Helpers ────────────────────────
def track_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = setup_logger()
        logger.info(f"Running '{func.__name__}'...")

        start_time = time.time()
        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()

        logger.info(f"'{func.__name__}' completed in {end_time - start_time:.4f} sec")
        logger.info(f"Memory used: {current / 1024:.2f} KB (peak: {peak / 1024:.2f} KB)")

        return result
    return wrapper


def load_yaml(path):
    logger = setup_logger(filename="logs")
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Loaded config from {path}")
        return cfg
    except CustomException as e:
        logger.error(f"Error loading config from {path}: {e}")
        

def write_yaml(path, config_dict):
    logger = setup_logger(filename="logs")
    try:
        with open(path, "w") as f:
            yaml.dump(config_dict, f)
        logger.info(f"Saved config to {path}")
    except CustomException as e:
        logger.critical(f"Error writing config to {path}: {e}")
        

def append_yaml(path, new_data_dict):
    logger = setup_logger(filename="logs")
    try:
        try:
            with open(path, "r") as f:
                existing_data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            existing_data = {}

        existing_data.update({k: float(v) for k, v in new_data_dict.items()})
        
        with open(path, "w") as f:
            yaml.dump(existing_data, f)
        logger.info(f"Appended and saved config to {path}")
    except CustomException as e:
        logger.critical(f"Error appending config to {path}: {e}")
    
def copy_csv_file(source_file: str, destination_folder: str):
    logger = setup_logger(filename="logs")
    try:
        src = Path(source_file)
        dest = Path(destination_folder)
        dest.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            raise FileNotFoundError(f"CSV file not found: {src}")
        if src.suffix != ".csv":
            raise ValueError("Provided file is not a .csv")

        dest_file = dest / src.name
        shutil.copy(src, dest_file)
        logger.info(f"Copied CSV: {src} to {dest_file}")
        print(f"Copied CSV: {src.name} to {dest_file}")

    except CustomException as e:
        logger.error(f"CSV copy failed: {e}")
        print(f"Error copying CSV: {e}")

def copy_yaml_file(source_file: str, destination_folder: str):
    logger = setup_logger(filename="logs")
    try:
        src = Path(source_file)
        dest = Path(destination_folder)
        dest.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            raise FileNotFoundError(f"YAML file not found: {src}")
        if src.suffix not in [".yaml", ".yml"]:
            raise ValueError("Provided file is not a .yaml or .yml")

        dest_file = dest / src.name
        shutil.copy(src, dest_file)
        logger.info(f"Copied YAML: {src} to {dest_file}")
        print(f"Copied YAML: {src.name} to {dest_file}")

    except CustomException as e:
        logger.error(f"YAML copy failed: {e}")
        print(f"Error copying YAML: {e}")

# ------------------ Copy All Joblib Files ------------------ #
def copy_selected_files(source_folder: str, destination_folder: str, extensions=(".joblib", ".yaml")):
    logger = setup_logger(filename="logs")
    try:
        src_folder = Path(source_folder)
        dest_folder = Path(destination_folder)
        dest_folder.mkdir(parents=True, exist_ok=True)

        if not src_folder.is_dir():
            raise ValueError(f"Source path is not a directory: {src_folder}")

        # Collect all matching files (e.g., .joblib and .yaml)
        matched_files = [file for ext in extensions for file in src_folder.rglob(f"*{ext}")]
        if not matched_files:
            raise FileNotFoundError(f"No files with extensions {extensions} found in {src_folder}")

        for file in matched_files:
            dest_file = dest_folder / file.name
            shutil.copy(file, dest_file)
            logger.info(f"Copied {file.suffix.upper()}: {file} → {dest_file}")
            print(f"Copied {file.name} → {dest_file}")

    except Exception as e:
        logger.error(f"File copy failed: {e}")
        print(f"Error copying files: {e}")


def copy_selected_files1(source_dir: str, destination_dir: str, file_types: List[str]) -> None:
   
    if not os.path.exists(source_dir):
        raise ValueError(f"Source directory does not exist: {source_dir}")
    
    os.makedirs(destination_dir, exist_ok=True)
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)

    for root, _, files in os.walk(source_dir):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in file_types):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(destination_dir, file)
                shutil.copy2(src_file, dest_file)
                print(f"Copied: {src_file} -> {dest_file}")


def delete_joblib_model(folder_path: str):
    logger = setup_logger(filename="logs")
    try:
        folder = Path(folder_path)
        if not folder.is_dir():
            raise ValueError(f"Provided path is not a directory: {folder}")

        joblib_files = list(folder.rglob("*.joblib"))
        if not joblib_files:
            print("No .joblib files found.")
            return

        for file in joblib_files:
            file.unlink()
            logger.info(f"Deleted: {file}")
            print(f"Deleted: {file.name}")

    except CustomException as e:
        logger.error(f"Error deleting joblib files: {e}")
    

def run_dvc_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def execute_files_backup():
    logger = setup_logger(filename="logs")
    try:
        logger.info("Starting Backing up of final_data.csv , joblib model , mlflow details yaml for previous trading day..")
        copy_selected_files1("Data/previous_data", "Data/ref_data", [".csv", ".joblib", ".yaml"])
        copy_selected_files("Data/new_data", "Data/previous_data")
        #copy_csv_file("Data/processed_data/final_data.csv", "Data/previous_data")
        #copy_selected_files("Tuned_Model/", "Data/previous_data")
        logger.info("Backing up of previous trading day artifacts  completed ..")
    
    except CustomException as e:
        logger.error(f"Error in backing up files: {e}")

# ─── Exports ──────────────────────────────────────
__all__ = ["setup_logger", "CustomException", "track_performance", "load_yaml", "write_yaml", "append_yaml", "copy_csv_file", "copy_selected_files", "delete_joblib_model", "run_dvc_command"]
