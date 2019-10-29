import pandas as pd
import numpy as np
import itertools
from tensorflow import keras

from .base_forecasters import Forecaster, MonteCarloForecaster


class Seq2SeqForecaster(Forecaster):
    def __init__(self, pred_steps: int, nn_width: int = 20, nn_depth: int = 2):
        super().__init__(pred_steps, nn_width, nn_depth)

    @staticmethod
    def _build_model(pred_steps: int, nn_width: int, nn_depth: int):
        return (
            [keras.layers.LSTM(nn_width, return_sequences=True, input_shape=[None, 1])]
            + [
                keras.layers.LSTM(nn_width, return_sequences=True)
                for _ in range(nn_depth - 1)
            ]
            + [keras.layers.TimeDistributed(keras.layers.Dense(pred_steps))]
        )


class Seq2SeqMonteCarloForecaster(MonteCarloForecaster):
    def __init__(
        self,
        pred_steps: int,
        nn_width: int = 20,
        nn_depth: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__(pred_steps, nn_width, nn_depth, dropout_rate)

    @staticmethod
    def _build_model(
        pred_steps: int, nn_width: int, nn_depth: int, dropout_rate: float
    ):
        return (
            [keras.layers.LSTM(nn_width, return_sequences=True, input_shape=[None, 1])]
            + list(
                itertools.chain.from_iterable(
                    [
                        (
                            keras.layers.LSTM(nn_width, return_sequences=True),
                            keras.layers.Dropout(rate=dropout_rate),
                        )
                        for _ in range(nn_depth - 1)
                    ]
                )
            )
            + [keras.layers.TimeDistributed(keras.layers.Dense(pred_steps))]
        )
