"""
Run from console:
"""
import sys
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
from nauto_zoo.preprocess.sensor.safer import SaferSensorPreprocessor
from nauto_zoo.models.safer import Safer
from sensor import sensor_pb2
import json
import boto3
import numpy as np

def test_safer_model():
    with open(str(Path("./test_data/cb07cec3da024592-17117a6e711cda05.json").resolve()), 'r') as fh:
        event_message = json.loads(fh.read())

    # read test sensor files into a com_rec
    sensor_pb_gzip = [
        "./test_data/cb07cec3da024592-17117a6e711cda05/sensor/1a3d9d1c7051c9e84c0458a96ae5e89ede6d6224",
        "./test_data/cb07cec3da024592-17117a6e711cda05/sensor/25e77d7226b6683d8fd96465b75d18cdc340505b",
        "./test_data/cb07cec3da024592-17117a6e711cda05/sensor/68ad8b85624814fcec908b0b33f1857350f7475e",
        "./test_data/cb07cec3da024592-17117a6e711cda05/sensor/89245cd0d7ff0dcf12038c42a0e4f5334b2f80d7",
        "./test_data/cb07cec3da024592-17117a6e711cda05/sensor/db71971c433da59301f899eea7faa0f557d785a6"]

    preprocessor = SaferSensorPreprocessor()
    preprocessor_output = preprocessor.preprocess_sensor_files(sensor_pb_gzip, event_message['event'])

    model_input = ModelInput(metadata=event_message['event'])
    model_input.set('sensor', preprocessor_output)

    config_provided_by_ucm = {'model_version': 0.1, 'threshold': 0.5}
    model = Safer(config_provided_by_ucm)
    model.set_logger(Mock())
    s3_client = boto3.resource('s3', region_name='us-east-1')
    model.set_s3(s3_client)
    model.bootstrap()
    response = model.run(model_input)

    assert response.summary == 'TRUE'
    assert response.confidence == 71