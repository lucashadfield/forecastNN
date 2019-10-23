import pandas as pd
import numpy as np
from tensorflow import keras


class Seq2SeqForecaster(keras.models.Sequential):
    def __init__(self, pred_steps: int, nn_width: int = 20, nn_depth: int = 2):
        self.pred_steps = pred_steps
        self.nn_width = nn_width
        self.nn_depth = nn_depth
        self.fit_history = []
        super().__init__(self._build_model())

    def _build_model(self):
        return (
            [
                keras.layers.LSTM(
                    self.nn_width, return_sequences=True, input_shape=[None, 1]
                )
            ]
            + [
                keras.layers.LSTM(self.nn_width, return_sequences=True)
                for _ in range(self.nn_depth - 1)
            ]
            + [keras.layers.TimeDistributed(keras.layers.Dense(self.pred_steps))]
        )

    def compile(self, loss='mse', optimizer='adam', **kwargs):
        super().compile(loss=loss, optimizer=optimizer, **kwargs)

    def _preprocess_df(self, df: pd.DataFrame, validation_split: float = 0.2):
        n_series, n_obs = df.shape
        n_steps = n_obs - self.pred_steps
        train_size = round(n_series * (1 - validation_split))

        series = df.values.reshape(*df.shape, 1)

        X_train = series[:train_size, :n_steps]
        X_valid = series[train_size:, :n_steps]

        Y = np.empty((n_series, n_steps, self.pred_steps))
        for step_ahead in range(1, 10 + 1):
            Y[..., step_ahead - 1] = series[..., step_ahead : step_ahead + n_steps, 0]

        Y_train = Y[:train_size]
        Y_valid = Y[train_size:]

        return X_train, X_valid, Y_train, Y_valid

    def fit_df(
        self,
        df: pd.DataFrame,
        epochs: int = 20,
        validation_split: float = 0.2,
        **kwargs
    ):
        X_train, Y_train, X_val, Y_val = self._preprocess_df(df, validation_split)

        history = super().fit(
            X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val), **kwargs
        )

        self.fit_history.append(history)

        return history

    def plot_history(self, history_step=0):
        plot_data = self.fit_history[-1 - history_step]
        # plot it
