from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Summary, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from pydantic import BaseModel
from typing import Dict
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime

from Common_Utils import CustomException, setup_logger, track_performance, load_yaml
from DataBase.db_handler import SQLiteEditor
from src.Data_Preprocessing import data_preprocessing_steps
from Model_Utils.feature_splitting_scaling import ScalingWithSplitStrategy

# ------------------- Init ------------------- #
app = FastAPI(title="Stock Index Predictor")
logger = setup_logger(filename="logs")

DB_PATH = "Data/data.db"
mlflow_data = load_yaml(path="Tuned_Model/mlflow_details.yaml")
project_root = Path(__file__).resolve().parent
model_path = project_root / mlflow_data["saved_model_path"]
model = joblib.load(model_path)

# ------------------ Prometheus Metrics ------------------ #
REQUEST_COUNT = Counter("predict_requests_total", "Total number of predict requests")
REQUEST_LATENCY = Summary("predict_latency_seconds", "Prediction latency in seconds")

# ------------------ Pydantic Models ------------------ #
class PredictionResult(BaseModel):
    date: str
    prediction: float
    ticker_values: Dict[str, float]

class ErrorResponse(BaseModel):
    error: str

class MetricsResponse(BaseModel):
    predict_requests_total: int
    predict_latency_seconds_count: int
    predict_latency_seconds_sum: float

# ------------------ Exception Handlers ------------------ #
@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    logger.error(f"CustomException: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error"}
    )

# ------------------ Data Loading & Preprocessing ------------------ #
def load_and_prepare_data(query: str = "SELECT * FROM final_data"):
    db = SQLiteEditor()
    df = db.fetch_df(query)

    if df.empty:
        raise CustomException("final_data table is empty")

    df_preprocess = data_preprocessing_steps(df)
    df_model = df_preprocess.drop(columns=['date'])
    splitter = ScalingWithSplitStrategy()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.apply(df_model)

    return df, X_test

# ------------------ /predict-last ------------------ #
@app.get("/predict-last", response_model=PredictionResult, responses={500: {"model": ErrorResponse}})
@REQUEST_LATENCY.time()
def predict_last():
    try:
        REQUEST_COUNT.inc()
        df, X_test = load_and_prepare_data()

        if X_test.shape[0] == 0:
            raise CustomException("No test data available after splitting")

        last_row = X_test.iloc[[-1]]
        prediction = float(model.predict(last_row)[0])

        last_unscaled_row = df.iloc[[-1]].drop(columns=["date"])
        date = df["date"].iloc[-1]
        ticker_values = {col: float(last_unscaled_row.iloc[0][col]) for col in last_unscaled_row.columns}

        return PredictionResult(
            date=datetime.today().strftime("%Y-%m-%d"),
            prediction=prediction,
            ticker_values=ticker_values
        )

    except Exception as e:
        raise CustomException(f"Prediction failed: {str(e)}")

# ------------------ /metrics (Prometheus Text) ------------------ #
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ------------------ /metrics-json (Swagger Friendly) ------------------ #
@app.get("/metrics-json", response_model=MetricsResponse, summary="Get Prometheus metrics in JSON format")
def metrics_json():
    return MetricsResponse(
        predict_requests_total=int(REQUEST_COUNT._value.get()),
        predict_latency_seconds_count=int(REQUEST_LATENCY._count.get()),
        predict_latency_seconds_sum=float(REQUEST_LATENCY._sum.get())
    )
