import tensorflow as tf
from nauto_zoo import ModelInput, ModelResponse
from nauto_zoo import Model, ModelInput, ModelResponse
import logging
import os
import numpy as np
import collections
from typing import Dict, Any

class BackupDetector(Model):
    DEFAULT_S3_MODEL_VERSION_DIR = "0.1"
    MODEL_FILES_FOLDER = "/tmp/visual_backup_detector/"
    VISUAL_BACKUP_MODEL_FILE = MODEL_FILES_FOLDER + "VISUAL_BACKUP_MODEL"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        assert "threshold" in self._config
        self._logger = logging.getLogger()
        self._logger.info('logger started')
        self.bootstrapped = False
        self._config = config
        self._try_load()

    def bootstrap(self):
        if not self.bootstrapped:
            model_dir = str(self._config.get("model_version", self.DEFAULT_S3_MODEL_VERSION_DIR))
            _ = self._download_from_s3
            os.makedirs(self.MODEL_FILES_FOLDER, exist_ok=True)
            _("nauto-cloud-models-test-us", "visual_backup_detector/" +
              model_dir+"/model.hdf5", self.VISUAL_BACKUP_MODEL_FILE)
            self._try_load()

    def _try_load(self):
        if os.path.isfile(self.VISUAL_BACKUP_MODEL_FILE):
            self.backup_model = tf.keras.models.load_model(self.VISUAL_BACKUP_MODEL_FILE)
            self._logger.info('Backup Model loaded...')
            self.bootstrapped = True

    def generate_judgement_from_preds(self, preds, backup_p99_treshold, time_buffer_threshold, iva_release_threshold):
        time_buffer_count = 0
        iva_triggerd = False
        iva_release = False
        iva_release_deque = collections.deque([0] * iva_release_threshold)

        for backup_score in preds:
            iva_release_deque.popleft()
            if backup_score >= backup_p99_treshold:
                time_buffer_count += 1
                iva_release_deque.append(0)
            else:
                iva_release_deque.append(1)
                if time_buffer_count > 0:
                    time_buffer_count -= 1

            iva_release_count = sum(iva_release_deque)
            if iva_triggerd and iva_release_count >= iva_release_threshold:
                iva_triggerd = False
                time_buffer_count = 0

                # trigger IVA
            if time_buffer_count >= time_buffer_threshold:
                iva_triggerd = True
                break
        return iva_triggerd

    def run(self, model_input: ModelInput) -> ModelResponse:
        assert self.bootstrapped  # Check if the model file is available
        frames = model_input.get('video-out')
        backup_p99_treshold = self._config.get('threshold', 0.5)
        preds = []
        for frame in frames:
            preds.append(self.backup_model.predict(np.expand_dims(frame, axis=0))[0, 1])
        is_backup = self.generate_judgement_from_preds(preds, backup_p99_treshold, 7.5, 7)
        return ModelResponse(summary=str(is_backup).upper(),
                             score=float(np.max(preds)),
                             confidence=100,
                             raw_output=[float(pred) for pred in preds]
                             )