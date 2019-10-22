import pandas as pd
from tensorflow import keras


class Seq2SeqForecaster:
    def __init__(
        self,
        df: pd.DataFrame,
        pred_timesteps: int,
        nn_width: int = 20,
        nn_depth: int = 2,
        dropout: float = 0.2,
        loss='mse',
        optimizer='adam',
        compile_kwargs=None,
    ):
        self.df = df
        self.pred_timesteps = pred_timesteps
        self.nn_width = nn_width
        self.nn_depth = nn_depth
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.compile_kwargs = compile_kwargs if compile_kwargs is not None else {}

        self.model = self._build_model()
        self._compile_model()

    def _build_model(self):
        return keras.models.Sequential(
            [
                keras.layers.LSTM(
                    self.nn_width, return_sequences=True, input_shape=[None, 1]
                )
            ]
            + [
                keras.layers.LSTM(self.nn_width, return_sequences=True)
                for _ in range(self.nn_depth - 1)
            ]
            + [keras.layers.TimeDistributed(keras.layers.Dense(self.pred_timesteps))]
        )

    def _compile_model(self):
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, **self.compile_kwargs
        )
