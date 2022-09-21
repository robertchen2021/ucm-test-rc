"""
Run from console:
"""

import logging
import os
import unittest
from pathlib import Path
from typing import List, Optional
from unittest.mock import Mock, patch

from nauto_datasets.core.sensors import CombinedRecording, Recording
from nauto_datasets.utils import protobuf
from nauto_zoo import ModelInput
from nauto_zoo.preprocess import SensorPreprocessorCombined, SensorPreprocessorImu
from nauto_zoo.preprocess.sensor.crashnet import CrashnetSensorPreprocessor
from nauto_zoo.models.irs import InstantaneousRiskPredictor
from sensor import sensor_pb2
import json
import boto3
import numpy as np

def test_irs_model():
    # assign event message
    with open(str(Path("./test_data/2e4afec04c5265f2-16b48fe9a33cfa5e.json").resolve()), 'r') as fh:
        event_message = json.loads(fh.read())

        # read test sensor files into a com_rec
        sensor_pb_gzip = [
            str(Path("./test_data/2e4afec04c5265f2-16b48fe9a33cfa5e/01d125794a189aa50b608bde8c7a18808972c04c").resolve()),
            str(Path("./test_data/2e4afec04c5265f2-16b48fe9a33cfa5e/05f165c9aa698b4ceccfefa8ca10b75f2daf769b").resolve()),
            str(Path("./test_data/2e4afec04c5265f2-16b48fe9a33cfa5e/28cfe7873dd6549f38473aa3504c745e6343d901").resolve()),
            str(Path("./test_data/2e4afec04c5265f2-16b48fe9a33cfa5e/5310d838470aea7f153d172c0902c03334d0fc1a").resolve()),
            str(Path("./test_data/2e4afec04c5265f2-16b48fe9a33cfa5e/571fb5076c26c39ed234da5b2e8fbf40e000df08").resolve()),
            str(Path("./test_data/2e4afec04c5265f2-16b48fe9a33cfa5e/5e5a17af203eabf51944342a5a72110c45fde1a5").resolve()),
            str(Path("./test_data/2e4afec04c5265f2-16b48fe9a33cfa5e/6d95d1fc9cdd283a14a3420d6c9943780a985ae5").resolve()),
            str(Path("./test_data/2e4afec04c5265f2-16b48fe9a33cfa5e/82e85b79d717de1afb01e178ec27001427135001").resolve()),
            str(Path("./test_data/2e4afec04c5265f2-16b48fe9a33cfa5e/891b9575d50fc255cc7adbba2f0557e43d52edbc").resolve()),
            str(Path("./test_data/2e4afec04c5265f2-16b48fe9a33cfa5e/90e9255e38d60f368e807b2a2c439072d59aeb89").resolve())
        ]
        config_provided_by_ucm = {'model_version': "0.1",
                                  'tf_xla': True}

        com_rec = SensorPreprocessorCombined().preprocess_sensor_files(sensor_pb_gzip)
        model_input = ModelInput(metadata=event_message)
        model_input.set('sensor', com_rec)

        model = InstantaneousRiskPredictor(config_provided_by_ucm)
        # model.manual_bootstrap()
        model.set_logger(Mock())
        s3_client = boto3.resource('s3', region_name='us-east-1')
        model.set_s3(s3_client)
        model.bootstrap()
        response = model.run(model_input)

        assert response.summary == 'TRUE'
        assert response.confidence == 51