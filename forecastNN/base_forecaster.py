import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


class Forecaster(keras.models.Sequential):
    figsize = (10, 5)

    def __init__(self, pred_steps, *args, **kwargs):
        super().__init__(self._build_model(pred_steps, *args, **kwargs))
        self.pred_steps = pred_steps
        self.fit_history = []

    @staticmethod
    def _build_model(*args, **kwargs):
        raise NotImplementedError

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

    def predict_df(self, df: pd.DataFrame, last_only: bool = True):
        pred_input = df.values.reshape(*df.shape, 1)
        preds = self.predict(pred_input)

        return preds[:, -1, :] if last_only else preds

    @staticmethod
    def _loss_df(history):
        num_epochs = len(history.history['loss'])

        return pd.DataFrame(history.history, index=range(1, num_epochs + 1)), num_epochs

    def plot_history(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        losses, fit_lengths = zip(
            *[self._loss_df(history) for history in self.fit_history]
        )

        plot_df = pd.concat(losses).reset_index(drop=True)
        plot_df.index += 1

        plot_df.plot(ax=ax, marker='.')

        for fit_marker in np.cumsum(fit_lengths)[:-1]:
            ax.axvline(fit_marker, zorder=-1, color='k', alpha=0.5, ls='dashed')

        ax.grid()

        return ax

    def plot_example_fit(self, df: pd.DataFrame, ax=None, loc: int = 0):
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        preds = self.predict_df(df.loc[[loc]], last_only=False)

        df.loc[loc].rename('Observations').plot(
            ax=ax, color='C1', alpha=0.8, marker='.', lw=3, legend=True
        )

        for i, row in pd.DataFrame(preds[0, :, :]).iterrows():
            tmp = row.copy()
            tmp.index += i + 1

            if i == len(preds[0]) - 1:
                tmp.rename('Future Prediction').plot(
                    ax=ax, color='C2', alpha=0.8, marker='.', lw=3, legend=True
                )
            elif i == len(preds[0]) - self.pred_steps - 1:
                tmp.rename('Validation Prediction').plot(
                    ax=ax, color='C0', alpha=0.8, marker='.', lw=3, legend=True
                )
            else:
                tmp.rename('Partial Predictions').plot(
                    ax=ax, color='k', alpha=0.2, legend=True if not i else False
                )

        ax.grid()

        return ax
