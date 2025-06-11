import yaml
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator
from typing import Union, Dict
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam, RMSprop


class ModelFactory:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_model(self, model_name: str, param_override: Dict = None) -> Union[BaseEstimator, tf.keras.Model]:
        if model_name not in self.config:
            raise ValueError(f"Model '{model_name}' not found in config")

        model_config = self.config[model_name]
        params = model_config['params']

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
            raise ValueError(f"Unsupported model: {model_name}")

    def build_lstm(self, params: Dict) -> tf.keras.Model:
        model = Sequential()
        input_shape = params.get("input_shape", (10, 1))  # default input shape

        lstm_layer = LSTM(
            units=params['units'],
            dropout=params['dropout'],
            recurrent_dropout=params['recurrent_dropout'],
            return_sequences=False,
            input_shape=input_shape
        )

        if params.get("bidirectional", False):
            model.add(Bidirectional(lstm_layer))
        else:
            model.add(lstm_layer)

        model.add(Dense(1))  # regression output

        # Optimizer selection
        optimizer_name = params['optimizer']
        lr = params['learning_rate']
        if optimizer_name == "adam":
            optimizer = Adam(learning_rate=lr)
        elif optimizer_name == "rmsprop":
            optimizer = RMSprop(learning_rate=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def train(self, model, X_train, y_train, model_name: str, X_val=None, y_val=None):
        if model_name == "lstm":
            params = self.config[model_name]['params']
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                verbose=1
            )
            return history
        else:
            model.fit(X_train, y_train)
            return model

    def predict(self, model, X_test, model_name: str):
        if model_name == "lstm":
            return model.predict(X_test).flatten()
        else:
            return model.predict(X_test)
