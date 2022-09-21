from nauto_zoo import Model, ModelInput, ModelResponse
from nauto_zoo.models.utils import infer_confidence
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from typing import Dict, Any, Optional, NamedTuple, Union
from .crashnet import Crashnet
import boto3
import logging
import os
# from data_reviewer.api import JUDGEMENT_SUMMARY_TRUE, JUDGEMENT_SUMMARY_FALSE

vehicle_profile_mapping_all = {	'LHD_DEFAULT':0,
    'LHD_CROSSOVER_SUV':1,
    'LHD_COMPACT':2,
    'LHD_FULL_SIZE':3,
    'LHD_MINIVAN':4,
    'LHD_PICKUP_TRUCK':5,
    'LHD_SEDAN':6,
    'LHD_SEMI':7,
    'LHD_SPORTS_CAR':8,
    'LHD_SUV':9,
    'LHD_TRUCK':10,
    'LHD_VAN':11,
    'RHD_DEFAULT':12,
    'RHD_COMPACT_CAR':13,
    'RHD_KEI_DEFAULT':14,
    'RHD_STANDARD_CAR':15,
    'RHD_TRUCK_TRAILER':16,
    'LHD_DEFAULT_N2': 0,
    'LHD_CROSSOVER_SUV_N2': 1,
    'LHD_COMPACT_N2': 2,
    'LHD_FULL_SIZE_N2': 3,
    'LHD_MINIVAN_N2': 4,
    'LHD_PICKUP_TRUCK_N2': 5,
    'LHD_SEDAN_N2': 6,
    'LHD_SEMI_N2': 7,
    'LHD_SPORTS_CAR_N2': 8,
    'LHD_SUV_N2': 9,
    'LHD_TRUCK_N2': 10,
    'LHD_VAN_N2': 11,
    'RHD_DEFAULT_N2': 12,
    'RHD_COMPACT_CAR_N2': 13,
    'RHD_KEI_DEFAULT_N2': 14,
    'RHD_STANDARD_CAR_N2': 15,
    'RHD_TRUCK_TRAILER_N2': 16,
    'LHD_DEFAULT_N3':0,
    'LHD_CROSSOVER_SUV_N3':1,
    'LHD_COMPACT_N3':2,
    'LHD_FULL_SIZE_N3':3,
    'LHD_MINIVAN_N3':4,
    'LHD_PICKUP_TRUCK_N3':5,
    'LHD_SEDAN_N3':6,
    'LHD_SEMI_N3':7,
    'LHD_SPORTS_CAR_N3':8,
    'LHD_SUV_N3':9,
    'LHD_TRUCK_N3':10,
    'LHD_VAN_N3':11,
    'RHD_DEFAULT_N3':12,
    'RHD_COMPACT_CAR_N3':13,
    'RHD_KEI_DEFAULT_N3':14,
    'RHD_STANDARD_CAR_N3':15,
    'RHD_TRUCK_TRAILER_N3':16 }


vehicle_profile_mapping_us = {	'LHD_DEFAULT':0,
    'LHD_CROSSOVER_SUV':1,
    'LHD_COMPACT':2,
    'LHD_FULL_SIZE':3,
    'LHD_MINIVAN':4,
    'LHD_PICKUP_TRUCK':5,
    'LHD_SEDAN':6,
    'LHD_SEMI':7,
    'LHD_SPORTS_CAR':8,
    'LHD_SUV':9,
    'LHD_TRUCK':10,
    'LHD_VAN':11,
    'RHD_DEFAULT':0,
    'RHD_COMPACT_CAR':0,
    'RHD_KEI_DEFAULT':0,
    'RHD_STANDARD_CAR':0,
    'RHD_TRUCK_TRAILER':0,
    'LHD_DEFAULT_N2': 0,
    'LHD_CROSSOVER_SUV_N2': 1,
    'LHD_COMPACT_N2': 2,
    'LHD_FULL_SIZE_N2': 3,
    'LHD_MINIVAN_N2': 4,
    'LHD_PICKUP_TRUCK_N2': 5,
    'LHD_SEDAN_N2': 6,
    'LHD_SEMI_N2': 7,
    'LHD_SPORTS_CAR_N2': 8,
    'LHD_SUV_N2': 9,
    'LHD_TRUCK_N2': 10,
    'LHD_VAN_N2': 11,
    'RHD_DEFAULT_N2': 0,
    'RHD_COMPACT_CAR_N2': 0,
    'RHD_KEI_DEFAULT_N2': 0,
    'RHD_STANDARD_CAR_N2': 0,
    'RHD_TRUCK_TRAILER_N2': 0,
    'LHD_DEFAULT_N3':0,
    'LHD_CROSSOVER_SUV_N3':1,
    'LHD_COMPACT_N3':2,
    'LHD_FULL_SIZE_N3':3,
    'LHD_MINIVAN_N3':4,
    'LHD_PICKUP_TRUCK_N3':5,
    'LHD_SEDAN_N3':6,
    'LHD_SEMI_N3':7,
    'LHD_SPORTS_CAR_N3':8,
    'LHD_SUV_N3':9,
    'LHD_TRUCK_N3':10,
    'LHD_VAN_N3':11,
    'RHD_DEFAULT_N3':0,
    'RHD_COMPACT_CAR_N3':0,
    'RHD_KEI_DEFAULT_N3':0,
    'RHD_STANDARD_CAR_N3':0,
    'RHD_TRUCK_TRAILER_N3':0 }


class CollisionDetector(Model):
    DEFAULT_S3_MODEL_VERSION_DIR = "0.1"
    MODEL_FILES_FOLDER = "/tmp/crashnet/"
    CRASHNET_MODEL_FILE = MODEL_FILES_FOLDER + "CRASHNET_MODEL"

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
            _("nauto-cloud-models-test-us", "crashnet_v15_ml_vt/"+model_dir+"/model.hdf5", self.CRASHNET_MODEL_FILE)
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
            _("nauto-cloud-models-test-us", "crashnet_v15_ml_vt/" + model_dir + "/model.hdf5", self.CRASHNET_MODEL_FILE)
            self._try_load()

    def _try_load(self):
        if os.path.isfile(self.CRASHNET_MODEL_FILE):
            run_config = RunConfig()
            self.crashnet_model = Crashnet(run_config, self.CRASHNET_MODEL_FILE)
            self.logger.info('crashnet model loaded')
            self.bootstrapped = True

    def run(self, model_input: ModelInput) -> ModelResponse:
        assert self.bootstrapped
        inputs = model_input.get('sensor')
        if type(inputs) is not list:
            inputs1 = [inputs]

        # Add the vehicle type in preprocessed sensor streams
        self.logger.info('event_type: %s'%model_input.metadata['event']['type'])
        if model_input.metadata['event']['type'] == 'crashnet':
            try:
                vehicle_profile_input = model_input.metadata['event']['params']['crashnet_data']['vehicle_profile_name']
            except:
                vehicle_profile_input = 'LHD_DEFAULT'
        elif model_input.metadata['event']['type'] == 'severe-g-event':
            try:
                vehicle_profile_input = model_input.metadata['event']['params']['maneuver_data']['threshold_file_info'].split('-')[1].upper()
            except:
                try:
                    vehicle_profile_input = model_input.metadata['event']['params']['abcc_data']['threshold_file_info'].split('-')[1].upper()
                except:
                    vehicle_profile_input = 'LHD_DEFAULT'
        else:
            vehicle_profile_input = 'LHD_DEFAULT'

        self.logger.info('Vehicle profile input: %s'%vehicle_profile_input)
        if self.num_vt < 20:
            inputs2 = [to_categorical(np.array([vehicle_profile_mapping_us[vehicle_profile_input]]), num_classes=self.num_vt)]
        else:
            inputs2 = [
                to_categorical(np.array([vehicle_profile_mapping_all[vehicle_profile_input]]), num_classes=self.num_vt)]
        raw_output = [
            np.squeeze(self.crashnet_model.model.predict([_input1, _input2], batch_size=1)).tolist()[1]
            for _input1, _input2 in zip(inputs1, inputs2)
        ]

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
