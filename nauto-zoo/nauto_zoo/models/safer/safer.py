import tensorflow as tf
from nauto_zoo import ModelInput, ModelResponse
from nauto_zoo.models.utils import infer_confidence
from nauto_zoo import Model, ModelInput, ModelResponse
import logging
import os
import numpy as np
import collections
from typing import Dict, Any

class Safer(Model):
    DEFAULT_S3_MODEL_VERSION_DIR = "0.1"
    MODEL_FILES_FOLDER = "/tmp/safer/"
    SAFER_MODEL_FILE = MODEL_FILES_FOLDER + "SAFER_MODEL"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._logger = logging.getLogger()
        self._logger.info('logger started')
        self.bootstrapped = False
        self._config = config
        self.model_dir = str(self._config.get("model_version", self.DEFAULT_S3_MODEL_VERSION_DIR))
        self.safer_threshold = self._config.get('threshold', 0.5)
        self._logger.info('Safer Model directory: %s' % self.model_dir)
        self._logger.info('Safer threhsold from config: %f' % self.safer_threshold)
        self._try_load()

    def bootstrap(self):
        if not self.bootstrapped:
            _ = self._download_from_s3
            os.makedirs(self.MODEL_FILES_FOLDER, exist_ok=True)
            _("nauto-cloud-models-test-us", "safer/" +
              self.model_dir+"/model.hdf5", self.SAFER_MODEL_FILE)
            self._try_load()

    def _try_load(self):
        if os.path.isfile(self.SAFER_MODEL_FILE):
            self.safer_model = tf.keras.models.load_model(self.SAFER_MODEL_FILE)
            self._logger.info('Safer Model v%s loaded...'%self.model_dir)
            self.bootstrapped = True

    def run(self, model_input: ModelInput) -> ModelResponse:
        assert self.bootstrapped  # Check if the model file is available
        preprocessor_output = model_input.get('sensor')
        safer_input, info = preprocessor_output

        pred = self.safer_model.predict(safer_input)
        return ModelResponse(summary=str(pred[0,0]>=self.safer_threshold).upper(),
                             score=float(pred[0,0]),
                             confidence=infer_confidence(pred[0,0], self.safer_threshold),
                             raw_output=info
                             )