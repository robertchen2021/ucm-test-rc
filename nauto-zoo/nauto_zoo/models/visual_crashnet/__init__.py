from nauto_zoo import Model, ModelInput, ModelResponse
from nauto_zoo.models.utils import infer_confidence
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from typing import Dict, Any, Optional, NamedTuple, Union
from .visual_crashnet import VisualCrashnet
import boto3
import logging
import os

class CollisionDetector(Model):
    DEFAULT_S3_MODEL_VERSION_DIR = "0.1"
    MODEL_FILES_FOLDER = "/tmp/visual_crashnet/"
    VISUAL_CRASHNET_MODEL_FILE = MODEL_FILES_FOLDER + "model_visual_crashnet.hdf5"
    DIST_MODEL_FILE = MODEL_FILES_FOLDER + "model_dist.hdf5"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        assert "threshold" in self._config
        self.logger = logging.getLogger()
        self.logger.info('logger started')
        self.bootstrapped = False
        self.num_vt = config.get('num_vt')
        self._try_load()

    def bootstrap(self):
        if not self.bootstrapped:
            model_dir =  str(self._config.get("model_version", self.DEFAULT_S3_MODEL_VERSION_DIR))

            _ = self._download_from_s3

            os.makedirs(self.MODEL_FILES_FOLDER, exist_ok=True)
            _("nauto-cloud-models-test-us", "visual_crashnet/" + model_dir + "/model_dist.hdf5", self.DIST_MODEL_FILE)

            os.makedirs(self.MODEL_FILES_FOLDER, exist_ok=True)
            _("nauto-cloud-models-test-us", "visual_crashnet/" + model_dir + "/model_visual_crashnet.hdf5", self.VISUAL_CRASHNET_MODEL_FILE)
            self._try_load()

    def manual_bootstrap(self):
        if not self.bootstrapped:
            app_logger = logging.getLogger('model')
            self.set_logger(app_logger)

            session = boto3.session.Session(profile_name='test-us')
            s3_client = session.resource('s3', region_name='us-east-1')
            self.set_s3(s3_client)

            # self.set_s3(boto3.resource('s3', region_name='us-east-1', endpoint_url=None))
            model_dir = str(self._config.get("model_version", self.DEFAULT_S3_MODEL_VERSION_DIR))
            _ = self._download_from_s3

            os.makedirs(self.MODEL_FILES_FOLDER, exist_ok=True)
            _("nauto-cloud-models-test-us", "visual_crashnet/" + model_dir + "/model_dist.hdf5", self.DIST_MODEL_FILE)

            os.makedirs(self.MODEL_FILES_FOLDER, exist_ok=True)
            _("nauto-cloud-models-test-us", "visual_crashnet/" + model_dir + "/model_visual_crashnet.hdf5", self.VISUAL_CRASHNET_MODEL_FILE)
            self._try_load()

    def _try_load(self):
        if os.path.isfile(self.DIST_MODEL_FILE) and os.path.isfile(self.VISUAL_CRASHNET_MODEL_FILE):
            run_config = RunConfig()
            self.visual_crashnet_model = VisualCrashnet(run_config, [self.DIST_MODEL_FILE, self.VISUAL_CRASHNET_MODEL_FILE])
            self.logger.info('distraction and visual crashnet model loaded')
            self.bootstrapped = True

    def run(self, model_input: ModelInput) -> ModelResponse:
        assert self.bootstrapped
        inputs = model_input.get('video_in')
        if type(inputs) is not list:
            inputs1 = [inputs]

        # Add the vehicle type in preprocessed sensor streams
        self.logger.info('event_type: %s'%model_input.metadata['event']['type'])
        embeddings = self.visual_crashnet_model.model_dist.predict(inputs1)
        prediction_score = self.visual_crashnet_model.model_visual_crashnet.predict(np.expand_dims(embeddings, axis=0))
        raw_output = np.squeeze(prediction_score).tolist()[1]

        response = ModelResponse(
            summary="TRUE" if np.max(raw_output) > self._config["threshold"] else "FALSE",
            score=np.max(raw_output).tolist(),
            confidence=infer_confidence(np.max(raw_output), self._config["threshold"]),
            raw_output={
                "score": raw_output,
                "threshold": self._config["threshold"]
            }
        )
        self.logger.info('message_id:%s, score:%f, threshold:%f, confidence: %d' % (model_input.metadata['event']['message_id'], np.max(raw_output),
        self._config["threshold"], infer_confidence(np.max(raw_output), self._config["threshold"])))

        return response

class RunConfig(NamedTuple):
    class_name: Optional[str] = None
    tf_xla: bool = True
    preferred: bool = False