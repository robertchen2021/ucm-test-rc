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
from nauto_zoo.models.crashnet import CollisionDetector
from sensor import sensor_pb2
import json
import boto3
import numpy as np

def to_categorical(y, num_classes=None, dtype='float32'):
  """Converts a class vector (integers) to binary class matrix.

  E.g. for use with categorical_crossentropy.

  Arguments:
      y: class vector to be converted into a matrix
          (integers from 0 to num_classes).
      num_classes: total number of classes. If `None`, this would be inferred
        as the (largest number in `y`) + 1.
      dtype: The data type expected by the input. Default: `'float32'`.

  Returns:
      A binary matrix representation of the input. The classes axis is placed
      last.

  Raises:
      Value Error: If input contains string value

  """
  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical


def test_crashnet_model():
    # assign event message
    with open(str(Path("./test_data/76d54e946eea68fa-16dc2cba08f7df35.json").resolve()), 'r') as fh:
        event_message = json.loads(fh.read())

	# read test sensor files into a com_rec
    sensor_pb_gzip = [
        str(Path("./test_data/76d54e946eea68fa-16dc2cba08f7df35/sensor/6ad98bf91bb7930d141603a27f996e8e2481312e").resolve()),
        str(Path("./test_data/76d54e946eea68fa-16dc2cba08f7df35/sensor/27f5c0686f10c4f4811b813544309571d4f911c6").resolve()),
        str(Path("./test_data/76d54e946eea68fa-16dc2cba08f7df35/sensor/86f7ed4b159568a8d68fa5d6a8e7f957eb76e53c").resolve()),
        str(Path("./test_data/76d54e946eea68fa-16dc2cba08f7df35/sensor/b5566e51fac81f46caf7c2147a26f841d5ec5521").resolve()),
        str(Path("./test_data/76d54e946eea68fa-16dc2cba08f7df35/sensor/d136707085011b9017c8130a3ec73b872c69237d").resolve())
    ]
    config_provided_by_ucm = {  'window_size': 4000,
                                'gyro_scaling_factor': 25,
                                'gyro_clip': 220.1702,
                                'num_vt': 15,
                                'use_oriented': False,
                                'threshold': 0.08,
                                'model_version': "0.2",
                                'tf_xla': True}

    model_input = ModelInput(metadata=event_message)
    preprocessor = CrashnetSensorPreprocessor(window_size=config_provided_by_ucm['window_size'], use_oriented=config_provided_by_ucm['use_oriented'])
    com_rec = preprocessor.preprocess_sensor_files(sensor_pb_gzip, event_message['event'])
    model_input.set('sensor', com_rec)
    model = CollisionDetector(config_provided_by_ucm)
    # model.manual_bootstrap()
    model.set_logger(Mock())
    s3_client = boto3.resource('s3', region_name='us-east-1')
    model.set_s3(s3_client)
    model.bootstrap()
    response = model.run(model_input)

    assert response.summary == 'FALSE'
    assert round(response.score, 3) == 0.068
    assert response.confidence == 15


if __name__ == '__main__':
    test_crashnet_model()