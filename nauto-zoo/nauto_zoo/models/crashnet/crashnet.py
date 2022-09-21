from tensorflow.keras.models import Model as KerasModel
from nauto_zoo import ModelInput, ModelResponse
from tensorflow.keras.models import load_model
import numpy as np
import os
from typing import Optional


class Crashnet(object):

    def __init__(self, config, model_path):
        # super().__init__(config)
        if config.tf_xla:
            os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
        self.model = self._load_model(model_path, output_layer='collision_output')



    @staticmethod
    def _load_model(model_path:str, output_layer: Optional[str] = None) -> KerasModel:
        model = load_model(model_path)
        if output_layer is None:
            return model
        return KerasModel(
            inputs=model.input,
            outputs=model.get_layer(output_layer).output
        )

    def run(self, imu_stream, vehicle_profile) -> np.array:
        pass