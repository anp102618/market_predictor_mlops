import yaml
import numpy as np
from typing import Union, Dict, Optional
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam, RMSprop


class ModelFactory:
    """
    Factory class to build and manage different machine learning and deep learning models.
    """

    def __init__(self, config_path: str):
        """
        Initialize the ModelFactory by loading the model configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config from {config_path}: {e}")

    def get_model(self, model_name: str, param_override: Optional[Dict] = None) -> Union[BaseEstimator, tf.keras.Model]:
        """
        Returns an instance of the specified model with optional parameter overrides.

        Args:
            model_name (str): Name of the model to instantiate.
            param_override (dict, optional): Parameters to override default config.

        Returns:
            A model instance of type scikit-learn or Keras.
        """
        if model_name not in self.config:
            raise ValueError(f"Model '{model_name}' not found in config")

        model_config = self.config[model_name]
        params = model_config.get('params', {}).copy()

        if param_override:
            params.update(param_override)

        if model_name == "lasso":
            return Lasso(**params)
        elif model_name == "ridge":
            return Ridge(**params)
        elif model_name == "xgboost":
            return XGBRegressor(**params)
        elif model_name == "lstm":
            return self.build_lstm(params)
        else:
            raise ValueError(f"Unsupported model: '{model_name}'")

    def build_lstm(self, params: Dict) -> tf.keras.Model:
        """
        Constructs a Keras LSTM or Bidirectional LSTM model.

        Args:
            params (Dict): Dictionary of LSTM parameters.

        Returns:
            tf.keras.Model: Compiled LSTM model.
        """
        model = Sequential()
        input_shape = params.get("input_shape", (10, 1))

        lstm_layer = LSTM(
            units=params.get('units', 64),
            dropout=params.get('dropout', 0.2),
            recurrent_dropout=params.get('recurrent_dropout', 0.2),
            return_sequences=False,
            input_shape=input_shape
        )

        if params.get("bidirectional", False):
            model.add(Bidirectional(lstm_layer))
        else:
            model.add(lstm_layer)

        model.add(Dense(1))  # Regression output layer

        optimizer_name = params.get('optimizer', 'adam').lower()
        lr = params.get('learning_rate', 0.001)

        if optimizer_name == "adam":
            optimizer = Adam(learning_rate=lr)
        elif optimizer_name == "rmsprop":
            optimizer = RMSprop(learning_rate=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def train(
        self,
        model: Union[BaseEstimator, tf.keras.Model],
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Union[BaseEstimator, tf.keras.callbacks.History]:
        """
        Trains the model on the provided dataset.

        Args:
            model: The model to be trained.
            X_train: Features for training.
            y_train: Targets for training.
            model_name: Name of the model.
            X_val: Features for validation (only for LSTM).
            y_val: Targets for validation (only for LSTM).

        Returns:
            Trained model or training history.
        """
        if model_name == "lstm":
            params = self.config[model_name].get('params', {})
            return model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                epochs=params.get('epochs', 10),
                batch_size=params.get('batch_size', 32),
                verbose=1
            )
        else:
            model.fit(X_train, y_train)
            return model

    def predict(
        self,
        model: Union[BaseEstimator, tf.keras.Model],
        X_test: np.ndarray,
        model_name: str
    ) -> np.ndarray:
        """
        Generates predictions from the trained model.

        Args:
            model: Trained model.
            X_test: Test features.
            model_name: Name of the model.

        Returns:
            Predictions as a numpy array.
        """
        if model_name == "lstm":
            return model.predict(X_test).flatten()
        else:
            return model.predict(X_test)
