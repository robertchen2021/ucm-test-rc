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


class ABCSanitizer(Model):
    """
    This is an ensemble model that includes Gravity Detector and the ML Atypical Model. 
    Ideally, this will filter out most of the blatant FP ABC events.
    """
    G = 9.81
    Z_STD = 1.796195820808912
    DEFAULT_S3_MODEL_VERSION_DIR = "0.1"

    # Gravity Model
    GRAVITY_MODEL_FILES_FOLDER = "/tmp/gravity_model/"
    GRAVITY_MODEL_FILE = GRAVITY_MODEL_FILES_FOLDER + "GRAVITY_MODEL.joblib"
    GRAVITY_VERSION = "0.1"
    # Atypical Model
    ML_ATYPICAL_MODEL_FILES_FOLDER = "/tmp/atypical_model/"
    ML_ATYPICAL_MODEL_FILE = ML_ATYPICAL_MODEL_FILES_FOLDER + "ATYPICAL_MODEL.joblib"
    ATYPICAL_VERSION = "0.1"

    PREPROCESS_VERSION = "0.1"
    base_features = ['pitch_angle', 'speed_mean', 'x_acc_mean', 'x_acc_var', 'x_angle',
                     'x_gyro_mean', 'x_gyro_var', 'y_acc_mean', 'y_acc_var', 'y_angle',
                     'y_gyro_mean', 'y_gyro_var', 'z_acc_mean', 'z_acc_var', 'z_angle',
                     'z_gyro_mean', 'z_gyro_var']

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
            gravity_model_dir = str(self._config.get("gravity_version",
                                                     self.DEFAULT_S3_MODEL_VERSION_DIR))
            atypical_model_dir = str(self._config.get("atypical_version",
                                                      self.DEFAULT_S3_MODEL_VERSION_DIR))
            _ = self._download_from_s3

            os.makedirs(self.GRAVITY_MODEL_FILES_FOLDER, exist_ok=True)
            _("nauto-cloud-models-test-us", "gravity_model/" +
              gravity_model_dir+"/model.joblib", self.GRAVITY_MODEL_FILE)

            os.makedirs(self.ML_ATYPICAL_MODEL_FILES_FOLDER, exist_ok=True)
            _("nauto-cloud-models-test-us", "atypical_model/" +
              atypical_model_dir+"/model.joblib", self.ML_ATYPICAL_MODEL_FILE)

            self._try_load()

    def _try_load(self):
        if os.path.isfile(self.GRAVITY_MODEL_FILE) and os.path.isfile(self.ML_ATYPICAL_MODEL_FILE):
            self.gravity_model = load(self.GRAVITY_MODEL_FILE)
            self.ml_atypical_model = load(self.ML_ATYPICAL_MODEL_FILE)

            self._logger.info('Gravity and Atypical Model loaded...')
            self.bootstrapped = True

    @staticmethod
    def _check_model_inputs(model_input: ModelInput) -> None:
        """Assert if the necessary fields are available in the inputs"""
        if 'message_id' not in model_input.metadata['event']:
            raise MalformedModelInputError(
                "'message_id' was not found in model_input.metadata['event']")

        if not model_input.metadata['event']['params'].get('maneuver_data'):
            raise MalformedModelInputError(
                'This is not a Maneuver Service Event')

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

    def run(self, model_input: ModelInput) -> ModelResponse:

        assert self.bootstrapped  # Check if the model file is available
        assert self._config.get('orient_sensor')  # Must have Pitch angle

        self._check_model_inputs(model_input)

        if not isinstance(model_input.get('sensor'), CombinedRecording):
            raise IncompleteInputMediaError('Sensor is not CombinedRecording.')
        else:
            message_id = model_input.metadata['event']['message_id']
            self._logger.info(f'[GravityModel + AtypicalModel] message_id: {message_id}, starting...')
            utc_basetime = int(model_input.metadata['event']['params']['utc_boot_time_ns']) + \
                int(model_input.metadata['event']
                    ['params']['utc_boot_time_offset_ns'])

            event_start_sensor_ns = utc_basetime + int(model_input.metadata['event']['params']['event_start_sensor_ns'])
            event_end_sensor_ns = utc_basetime + int(model_input.metadata['event']['params']['event_end_sensor_ns'])
            event_end_sensor_ns = min(event_end_sensor_ns, event_start_sensor_ns + 5e9)
            features = PreprocessIMU(com_rec=model_input.get('sensor'),
                                     start_ns=event_start_sensor_ns,
                                     end_ns=event_end_sensor_ns,
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

            features['version'] = self._config.get('model_version')
            features['preprocess_version'] = self.PREPROCESS_VERSION
            features['event_duration_ms'] = int(
                model_input.metadata['event']['params']['maneuver_data']['event_duration_ns']) / 1e6
            features['event_type'] = model_input.metadata['event']['type']
            gravity_ft_vector = np.array([features['event_duration_ms'],
                                          features['pitch_angle'],
                                          (self.G -
                                           features['z_acc_mean']) / self.Z_STD,
                                          features['z_acc_var']
                                          ])[None]
            atypical_pred = self.ml_atypical_model.predict_proba(
                np.array([features[x] for x in self.base_features])[None])
            gravity_pred = self.gravity_model.predict_proba(gravity_ft_vector)

            result = np.average(
                [atypical_pred, gravity_pred], axis=0, weights=None)
            summary = bool(1 - np.argmax(result))

            self._logger.info(f'AbcSanitizer, message_id: {message_id}, summary: {summary}')
            return ModelResponse(summary=str(summary).upper(),
                                 score=float(np.max(result)),
                                 confidence=int(np.max(result) * 100),
                                 raw_output=features
                                 )
