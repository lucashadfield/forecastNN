import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


class Seq2SeqForecaster(keras.models.Sequential):
    def __init__(self, pred_steps: int, nn_width: int = 20, nn_depth: int = 2):
        super().__init__(self._build_model(pred_steps, nn_width, nn_depth))
        self.pred_steps = pred_steps
        self.fit_history = []

    @staticmethod
    def _build_model(pred_steps, nn_width, nn_depth):
        return (
            [keras.layers.LSTM(nn_width, return_sequences=True, input_shape=[None, 1])]
            + [
                keras.layers.LSTM(nn_width, return_sequences=True)
                for _ in range(nn_depth - 1)
            ]
            + [keras.layers.TimeDistributed(keras.layers.Dense(pred_steps))]
        )

    def compile(self, loss='mse', optimizer='adam', **kwargs):
        super().compile(loss=loss, optimizer=optimizer, **kwargs)

    def _preprocess_df(self, df: pd.DataFrame, validation_split: float = 0.2):
        n_series, n_obs = df.shape
        n_steps = n_obs - self.pred_steps
        train_size = round(n_series * (1 - validation_split))

        series = df.values.reshape(*df.shape, 1)

        X_train = series[:train_size, :n_steps]
        X_val = series[train_size:, :n_steps]

        Y = np.empty((n_series, n_steps, self.pred_steps))
        for step_ahead in range(1, 10 + 1):
            Y[..., step_ahead - 1] = series[..., step_ahead : step_ahead + n_steps, 0]

        Y_train = Y[:train_size]
        Y_val = Y[train_size:]

        return X_train, Y_train, X_val, Y_val

    def fit_df(
        self, df: pd.DataFrame, epochs: int, validation_split: float = 0.2, **kwargs
    ):
        X_train, Y_train, X_val, Y_val = self._preprocess_df(df, validation_split)

        history = super().fit(
            X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val), **kwargs
        )

        self.fit_history.append(history)

        return history

    @staticmethod
    def _loss_df(history):
        num_epochs = len(history.history['loss'])

        return pd.DataFrame(history.history, index=range(1, num_epochs + 1)), num_epochs

    def plot_history(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        losses, fit_lengths = zip(
            *[self._loss_df(history) for history in self.fit_history]
        )

        plot_df = pd.concat(losses).reset_index(drop=True)
        plot_df.index += 1

        plot_df.plot(ax=ax, marker='.')

        for fit_marker in np.cumsum(fit_lengths)[:-1]:
            ax.axvline(fit_marker, zorder=-1, color='k', alpha=0.5, ls='dashed')

        return ax
