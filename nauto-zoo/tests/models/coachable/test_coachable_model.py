"""
Run from console:
$ cd nauto-ai/nauto-zoo
$ python -m pytest tests/models/coachable/test_coachable_model.py --log-cli-level=INFO
"""

import json
import os
import unittest
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from unittest.mock import Mock

from nauto_zoo import ModelInput
from nauto_zoo.models.coachable_v2.coachable_model import CoachableModel
from nauto_zoo.preprocess import SensorPreprocessorCombined, SensorCoachablePreprocessor


class TestCoachableModel(unittest.TestCase):

    def test_coachable_v3_model_on_braking_hard_event_v4_should_process(self):
        # test the coachable v3 model
        # The coachable_v3 is a DL model, completely different from coachable and coachable_v2, which are python classes

        pwd = Path(__file__).resolve().parent

        # assign event message
        with open(os.path.join(pwd, "hard_brake_v4.json"), 'r') as fh:
            event_message = json.loads(fh.read())

        # read test sensor files and preprocess into the input format required by the coachable v3 DL model
        protobuf_dir = '1660ad11b5a2299a'
        protobuf_files = ['sensor_0.gz', 'sensor_1.gz', 'sensor_2.gz']
        sensor_pb_gzip = [os.path.join(pwd, protobuf_dir, pb) for pb in protobuf_files]
        preprocessor = SensorCoachablePreprocessor()
        preprocessed_data = preprocessor.preprocess_sensor_files(sensor_pb_gzip)

        # assemble the model input
        model_input = ModelInput(metadata=event_message)
        model_input.set('sensor', preprocessed_data)

        # load the compiled model
        model_path = os.path.join(pwd, 'coachable_v3_model_0.1.h5')
        model = load_model(model_path)

        # run inference using the style similar to the universal_model/model/keras_v2.py
        inputs = model_input.get('sensor')
        if type(inputs) is not list:
            inputs = [inputs]
        raw_output = [
            np.squeeze(model.predict(_input, batch_size=1)).tolist()[1]
            for _input in inputs
        ]

        # test the output
        self.assertEqual(len(raw_output), 1)
        self.assertTrue(raw_output[0] >= 0)
        self.assertTrue(raw_output[0] <= 1)

    def test_coachable_v3_model_on_risk_braking_event_should_process(self):
        # test the coachable v3 model
        # The coachable_v3 is a DL model, completely different from coachable and coachable_v2, which are python classes

        pwd = Path(__file__).resolve().parent

        # assign event message
        with open(os.path.join(pwd, "risk_BRAKING.json"), 'r') as fh:
            event_message = json.loads(fh.read())

        # read test sensor files and preprocess into the input format required by the coachable v3 DL model
        protobuf_dir = '1660ad11b5a2299a'
        protobuf_files = ['sensor_0.gz', 'sensor_1.gz', 'sensor_2.gz']
        sensor_pb_gzip = [os.path.join(pwd, protobuf_dir, pb) for pb in protobuf_files]
        preprocessor = SensorCoachablePreprocessor()
        preprocessed_data = preprocessor.preprocess_sensor_files(sensor_pb_gzip)

        # assemble the model input
        model_input = ModelInput(metadata=event_message)
        model_input.set('sensor', preprocessed_data)

        # load the compiled model
        model_path = os.path.join(pwd, 'coachable_v3_model_0.1.h5')
        model = load_model(model_path)

        # run inference using the style similar to the universal_model/model/keras_v2.py
        inputs = model_input.get('sensor')
        if type(inputs) is not list:
            inputs = [inputs]
        raw_output = [
            np.squeeze(model.predict(_input, batch_size=1)).tolist()[1]
            for _input in inputs
        ]

        # test the output
        self.assertEqual(len(raw_output), 1)
        self.assertTrue(raw_output[0] >= 0)
        self.assertTrue(raw_output[0] <= 1)

    def test_coachable_v2_model_on_braking_hard_event_v4_should_return_false(self):
        # both sensor files and json file are from config_version=3.4

        pwd = Path(__file__).resolve().parent

        with open(os.path.join(pwd, "hard_brake_v4.json"), 'r') as fh:
            event_message = json.loads(fh.read())

        protobuf_dir = '1660ad11b5a2299a'
        protobuf_files = ['sensor_0.gz', 'sensor_1.gz', 'sensor_2.gz']
        sensor_pb_gzip = [os.path.join(pwd, protobuf_dir, pb) for pb in protobuf_files]
        preprocessor = SensorPreprocessorCombined()
        com_rec = preprocessor.preprocess_sensor_files(sensor_pb_gzip)

        model_input = ModelInput(metadata=event_message)
        model_input.set('sensor', com_rec)
        config_provided_by_ucm = {'turn_time_th': 0.5,
                                  'tg_score_th': -1,
                                  'tg_duration_th': 0.5,
                                  'distraction_score_th': 0.5,
                                  'distraction_duration_th': 1.0,
                                  'holding_object_score_th': 0.5,
                                  'holding_object_duration_th': 1.0,
                                  'no_face_score_th': 0.5,
                                  'no_face_duration_th': 1.0,
                                  'seq_time_delta_th': 5.0,
                                  'startle_slope_th': 3.25,
                                  'startle_a_max_th': -6.25,
                                  'startle_a_mid_th': -3.,
                                  'startle_buffet_length': 200}
        model = CoachableModel(config_provided_by_ucm)
        model.set_logger(Mock())

        response = model.run(model_input)

        self.assertEqual(response.summary, 'FALSE')
        self.assertEqual(response.score, 0.)
        self.assertEqual(response.confidence, 100)
        self.assertIsNotNone(response.summary)

    def test_coachable_v2_model_on_braking_hard_event_v3_should_return_true(self):
        # both sensor files and json file are from config_version=3.3

        pwd = Path(__file__).resolve().parent
        with open(os.path.join(pwd, "hard_brake_v3.json"), 'r') as fh:
            event_message = json.loads(fh.read())

        protobuf_dir = '16513c82895dad08'
        protobuf_files = ['sensor_0.gz', 'sensor_1.gz', 'sensor_2.gz', 'sensor_3.gz']
        sensor_pb_gzip = [os.path.join(pwd, protobuf_dir, pb) for pb in protobuf_files]
        preprocessor = SensorPreprocessorCombined()
        com_rec = preprocessor.preprocess_sensor_files(sensor_pb_gzip)

        model_input = ModelInput(metadata=event_message)
        model_input.set('sensor', com_rec)
        config_provided_by_ucm = {'turn_time_th': 0.5,
                                  'tg_score_th': -1,
                                  'tg_duration_th': 0.5,
                                  'distraction_score_th': 0.5,
                                  'distraction_duration_th': 1.0,
                                  'holding_object_score_th': 0.5,
                                  'holding_object_duration_th': 1.0,
                                  'no_face_score_th': 0.5,
                                  'no_face_duration_th': 1.0,
                                  'seq_time_delta_th': 5.0,
                                  'startle_slope_th': 3.25,
                                  'startle_a_max_th': -6.25,
                                  'startle_a_mid_th': -3.,
                                  'startle_buffet_length': 200}
        model = CoachableModel(config_provided_by_ucm)
        model.set_logger(Mock())

        response = model.run(model_input)

        self.assertEqual(response.summary, 'TRUE')
        self.assertEqual(response.score, 1.)
        self.assertEqual(response.confidence, 100)
        self.assertIsNotNone(response.summary)

    def test_coachable_v2_model_on_risk_braking_event_should_return_false(self):
        # sensor files are from config_version=3.4
        # event json file is from a risk-BRAKING event from config_version=3.7

        pwd = Path(__file__).resolve().parent

        with open(os.path.join(pwd, "risk_BRAKING.json"), 'r') as fh:
            event_message = json.loads(fh.read())

        protobuf_dir = '1660ad11b5a2299a'
        protobuf_files = ['sensor_0.gz', 'sensor_1.gz', 'sensor_2.gz']
        sensor_pb_gzip = [os.path.join(pwd, protobuf_dir, pb) for pb in protobuf_files]
        preprocessor = SensorPreprocessorCombined()
        com_rec = preprocessor.preprocess_sensor_files(sensor_pb_gzip)

        model_input = ModelInput(metadata=event_message)
        model_input.set('sensor', com_rec)
        config_provided_by_ucm = {'turn_time_th': 0.5,
                                  'tg_score_th': -1,
                                  'tg_duration_th': 0.5,
                                  'distraction_score_th': 0.5,
                                  'distraction_duration_th': 1.0,
                                  'holding_object_score_th': 0.5,
                                  'holding_object_duration_th': 1.0,
                                  'no_face_score_th': 0.5,
                                  'no_face_duration_th': 1.0,
                                  'seq_time_delta_th': 5.0,
                                  'startle_slope_th': 3.25,
                                  'startle_a_max_th': -6.25,
                                  'startle_a_mid_th': -3.,
                                  'startle_buffet_length': 200}
        model = CoachableModel(config_provided_by_ucm)
        model.set_logger(Mock())

        response = model.run(model_input)

        self.assertEqual(response.summary, 'FALSE')
        self.assertEqual(response.score, 0.)
        self.assertEqual(response.confidence, 100)
        self.assertIsNotNone(response.summary)


if __name__ == '__main__':
    unittest.main()

