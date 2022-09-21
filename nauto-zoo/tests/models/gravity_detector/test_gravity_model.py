"""
Run from Console
$ cd nauto-ai/nauto-zoo
$ python -m pytest tests/models/gravity_detector/test_gravity_detector.py --log-cli-level=INFO
"""
# Prod-us-console (AWS S3 -- cloud formation/watch)
import logging
import os
import unittest
from pathlib import Path
from typing import List, Optional
import boto3
import json
from unittest.mock import Mock, patch

from nauto_datasets.core.sensors import CombinedRecording, Recording
from nauto_datasets.utils import protobuf
from nauto_zoo import ModelInput

from nauto_zoo.preprocess import SensorPreprocessorCombined
from nauto_zoo.models.gravity_detector.gravity_detector import GravityDetector


class TestGravityModel(unittest.TestCase):
    """Unit Test for Gravity Model"""

    def test_gravity_model_v1(self):
        """Positive Test Case for Gravity Detector"""
        # drt_link: https://review.nauto.systems/event/_/958be1a2-2418-4dc8-8808-cdd34d846494-us-east-1/_/_/_
        pwd = Path(__file__).resolve().parent
        test_data_path = "../../../test_data/gravity_detector_test_data/positive_case"

        with open(os.path.join(pwd, test_data_path, "pos_case.json"), 'r') as fh:
            event_message = json.loads(fh.read())

        sensor_pb_gzip = ['9f6a681c6729361b0ad45e587434d383755b811c',
                          'b63c5ec9feb3cf296ef9909c055b68321f402dff',
                          'cec2b2e5f1f294cf74ae7c5bceaf16ad5ec13cc7']
        sensor_pb_gzip = [os.path.join(pwd, test_data_path, x)
                          for x in sensor_pb_gzip]
        com_rec = SensorPreprocessorCombined().preprocess_sensor_files(sensor_pb_gzip)
        model_input = ModelInput(metadata=event_message)
        model_input.set('sensor', com_rec)

        config_provided_by_ucm = {"window_length": 5,
                                  "acc_field": "rt_acc",
                                  "gyro_field": "rt_gyro",
                                  "gps_field": "gps",
                                  "to_utc_time": True,
                                  "imu_components": ["sensor_ns", "x", "y", "z"],
                                  "gps_components": ["sensor_ns", "speed"],
                                  "orient_sensor": True,
                                  "model_version": "0.1"
                                  }
        model = GravityDetector(config_provided_by_ucm)

        model.set_logger(Mock())
        s3_client = boto3.resource("s3", region_name="us-east-1")
        model.set_s3(s3_client)
        model.bootstrap()
        response = model.run(model_input)

        self.assertEqual(response.summary, 'FALSE')
        self.assertGreater(response.confidence, 95)
        self.assertGreater(response.score, .95)
        self.assertIsNotNone(response.summary)
        self.assertIsNotNone(response.raw_output)

    def test_gravity_model_v2(self):
        """Negative Test Case for Gravity Detector"""
        # drt_link: https://review.nauto.systems/event/_/0ff4da67-f674-420e-b84e-54b535790916-us-east-1/_/_/_
        pwd = Path(__file__).resolve().parent
        test_data_path = "../../../test_data/gravity_detector_test_data/negative_case"
        with open(os.path.join(pwd, test_data_path, "neg_case.json"), 'r') as fh:
            event_message = json.loads(fh.read())

        sensor_pb_gzip = ['4d67e38ef320ed5abf737d4ed2eb19da1963f56c',
                          '703dfb95f686c8254d0be39cee060487aa1bf54d',
                          '72a41a667c0eb516a04063a08c6df024b8a0fb78']
        sensor_pb_gzip = [os.path.join(pwd, test_data_path, x)
                          for x in sensor_pb_gzip]
        com_rec = SensorPreprocessorCombined().preprocess_sensor_files(sensor_pb_gzip)
        model_input = ModelInput(metadata=event_message)
        model_input.set('sensor', com_rec)

        config_provided_by_ucm = {"window_length": 5,
                                  "acc_field": "rt_acc",
                                  "gyro_field": "rt_gyro",
                                  "gps_field": "gps",
                                  "to_utc_time": True,
                                  "imu_components": ["sensor_ns", "x", "y", "z"],
                                  "gps_components": ["sensor_ns", "speed"],
                                  "orient_sensor": True,
                                  "model_version": "0.1"
                                  }
        model = GravityDetector(config_provided_by_ucm)
        model.set_logger(Mock())
        s3_client = boto3.resource("s3", region_name="us-east-1")
        model.set_s3(s3_client)
        model.bootstrap()
        response = model.run(model_input)

        self.assertEqual(response.summary, 'TRUE')
        self.assertGreater(response.score, 0.95)
        self.assertGreater(response.confidence, 95)
        self.assertIsNotNone(response.summary)


if __name__ == '__main__':
    unittest.main()
