from tensorflow.keras.models import Model as KerasModel
from nauto_zoo import ModelInput, ModelResponse
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import numpy as np
import os
from typing import Optional


class FusionCrashnet(object):
    def __init__(self, config, model_paths):
        if config.tf_xla:
            os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
        self.model_dist = self._load_model(model_paths[0], output_layer='global_average_pooling2d')
        self.model_fusion_crashnet = self._load_model(model_paths[1], output_layer='collision_output', custom_objects={'loss': tfa.losses.SigmoidFocalCrossEntropy(alpha=0.025, gamma=4.0)})

    @staticmethod
    def _load_model(model_path:str, output_layer: Optional[str] = None, custom_objects = None) -> KerasModel:
        if custom_objects is not None:
            model = load_model(model_path, custom_objects=custom_objects)
        else:
            model = load_model(model_path)
        if output_layer is None:
            return model
        return KerasModel(
            inputs=model.input,
            outputs=model.get_layer(output_layer).output
        )

    def run(self, imu_stream, vehicle_profile) -> np.array:
        pass
