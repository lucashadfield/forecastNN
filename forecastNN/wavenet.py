from tensorflow import keras
from .base_forecaster import Forecaster


class WaveNetForecaster(Forecaster):
    def __init__(
        self,
        pred_steps: int,
        layer_steps: int = 4,
        n_filters: int = 20,
        activation: str = 'relu',
    ):
        super().__init__(pred_steps, layer_steps, n_filters, activation)

    @staticmethod
    def _build_model(pred_steps, layer_steps, n_filters, activation):
        return (
            [keras.layers.InputLayer(input_shape=[None, 1])]
            + [
                keras.layers.Conv1D(
                    filters=n_filters,
                    kernel_size=2,
                    padding='causal',
                    activation=activation,
                    dilation_rate=rate,
                )
                for rate in [2 ** i for i in range(layer_steps)] * 2
            ]
            + [keras.layers.Conv1D(filters=pred_steps, kernel_size=1)]
        )
