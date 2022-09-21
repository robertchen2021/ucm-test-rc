import numpy as np
import os
from joblib import load
from typing import Dict, Any
import json

from nauto_zoo import (Model, ModelResponse, ModelInput, IncompleteInputMediaError,
                       MalformedModelInputError)

from nauto_datasets.core.sensors import CombinedRecording
from .sensor_preprocessor import PreprocessIMU
import logging


class GravityDetector(Model):
    """
    In this coachable model, we will need 
    1. CombinedRecording
    2. EventStartNs
    3. EventEndNs
    4. Features: ['event_duration_ms', 'pitch_angle', 'std_residual_avg', 'z_acc_var']
    5. Message Types: acceleration-hard, braking-hard, corner-left-hard, corner-right-hard
    Constants: 
    G: Earth Surface Gravitational Acceleration
    Z_STD: Training standard deviation for all 
    # S3 Directory: nauto-cloud-models-test-us/gravity_model/0.1/model.joblib
    # Local Directory /tmp/gravity_model/GRAVITY_MODEL.joblib
    """
    G = 9.81
    Z_STD = 1.796195820808912  # This could be part of the config
    z_acc_stds = {'acceleration-hard': 2.950004812178165,
                  'braking-hard': 0.8883773878022703,
                  'corner-left-hard': 0.5254777813589878,
                  'corner-right-hard': 0.6350808490712344}
    DEFAULT_S3_MODEL_VERSION_DIR = "0.1"
    MODEL_FILES_FOLDER = "/tmp/gravity_model/"
    GRAVITY_MODEL_FILE = MODEL_FILES_FOLDER + "GRAVITY_MODEL.joblib"
    PREPROCESS_VERSION = "0.1"

    def __init__(self, config: Dict[str, Any]):
        # Create logger
        super().__init__(config)
        self.logger = logging.getLogger()
        self.set_logger(logging.getLogger())
        self._logger.info('Logger Started...')
        self.bootstrapped = False
        self._config = config
        self._try_load()

    def bootstrap(self):
        if not self.bootstrapped:
            model_dir = str(self._config.get("model_version",
                                             self.DEFAULT_S3_MODEL_VERSION_DIR))
            _ = self._download_from_s3
            os.makedirs(self.MODEL_FILES_FOLDER, exist_ok=True)
            _("nauto-cloud-models-test-us", "gravity_model/" +
              model_dir+"/model.joblib", self.GRAVITY_MODEL_FILE)
            self._try_load()

    def _try_load(self):
        if os.path.isfile(self.GRAVITY_MODEL_FILE):
            self.gravity_model = load(self.GRAVITY_MODEL_FILE)
            self._logger.info('Gravity Model loaded...')
            self.bootstrapped = True

    def run(self, model_input: ModelInput) -> ModelResponse:
        assert self.bootstrapped  # Check if the model file is available
        assert self._config.get('orient_sensor')  # Must have Pitch angle
        if 'message_id' not in model_input.metadata['event']:
            raise MalformedModelInputError(
                "'message_id' was not found in model_input.metadata['event']")
        if not model_input.metadata['event']['params'].get('maneuver_data'):
            raise MalformedModelInputError(
                'This is not a Maneuver Serrvice Event')
        if not isinstance(int(model_input.metadata['event']['params'].get('event_start_sensor_ns')), int):
            tmp = model_input.metadata['event']['params'].get(
                'event_start_sensor_ns')
            raise MalformedModelInputError(f"event_start_sensor_ns is missing! dtype={type(tmp)}, value={tmp}")

        if not isinstance(int(model_input.metadata['event']['params'].get('event_end_sensor_ns')), int):
            tmp = model_input.metadata['event']['params'].get(
                'event_end_sensor_ns')
            raise MalformedModelInputError(f"event_end_sensor_ns is missing! dtype={type(tmp)}, value={tmp}")

        if not isinstance(int(model_input.metadata['event']['params']['maneuver_data'].get('event_duration_ns')), int):
            tmp = model_input.metadata['event']['params']['maneuver_data'].get(
                'event_duration_ns')
            raise MalformedModelInputError(
                f"Maneuver Data event_duration_ns is missing! dtype={type(tmp)}, value={tmp}")

        if not model_input.get('sensor'):
            raise IncompleteInputMediaError(
                'No sensor CombinedRecording was given in the model_input.')
        if not isinstance(model_input.get('sensor'), CombinedRecording):
            raise IncompleteInputMediaError('Sensor is not CombinedRecording.')
        else:
            message_id = model_input.metadata['event']['message_id']
            self._logger.info(f'[GravityModel] message_id: {message_id}, starting...')
            utc_basetime = int(model_input.metadata['event']['params']['utc_boot_time_ns']) + \
                int(model_input.metadata['event']
                    ['params']['utc_boot_time_offset_ns'])

            features = PreprocessIMU(com_rec=model_input.get('sensor'),
                                     start_ns=utc_basetime +
                                     int(model_input.metadata['event']
                                         ['params']['event_start_sensor_ns']),
                                     end_ns=utc_basetime +
                                     int(model_input.metadata['event']
                                         ['params']['event_end_sensor_ns']),
                                     acc_field=self._config.get('acc_field'),
                                     gyro_field=self._config.get('gyro_field'),
                                     gps_field=self._config.get('gps_field'),
                                     orient_sensor=self._config.get(
                                         'orient_sensor'),
                                     to_utc_time=self._config.get(
                                         'to_utc_time'),
                                     window_length=self._config.get(
                                         'window_length')
                                     ).to_feature_vector()

            features['threshold_file'] = model_input.metadata['event']['params']['maneuver_data'].get(
                'threshold_file_info')
            features['version'] = self._config.get('model_version')
            features['preprocess_version'] = self.PREPROCESS_VERSION
            features['event_duration_ms'] = int(
                model_input.metadata['event']['params']['maneuver_data']['event_duration_ns']) / 1e6
            features['event_type'] = model_input.metadata['event']['type']
            feature_vector = np.array([features['event_duration_ms'],
                                       features['pitch_angle'],
                                       (self.G -
                                        features['z_acc_mean']) / self.z_acc_stds.get(model_input.metadata['event']['type'],
                                                                                      self.Z_STD),
                                       features['z_acc_var']
                                       ])[None]
            result = self.gravity_model.predict_proba(feature_vector)
            summary = False if np.argmax(result) else True
            self._logger.info(f'GravityModel, message_id: {message_id}, summary: {summary}')
            return ModelResponse(summary=str(summary).upper(),
                                 score=float(np.max(result)),
                                 confidence=int(np.max(result) * 100),
                                 raw_output=features
                                 )
