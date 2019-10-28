from tensorflow import keras

from .base_forecaster import Forecaster


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
