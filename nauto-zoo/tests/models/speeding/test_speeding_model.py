"""
Run from console:
$ cd nauto-ai/nauto-zoo
$ python -m pytest tests/models/speeding/ --log-cli-level=INFO
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
from nauto_zoo.models.speeding import SpeedingModel
from sensor import sensor_pb2


def get_combined_recording(paths: List[str]) -> Optional['CombinedRecording']:
    """
    :param paths:
    :return:
    """

    if None in paths:
        logging.error('None path in sensor paths')
        return None

    try:
        gzip_files = [Path(p) for p in paths]
        recordings = []
        for file_path in gzip_files:
            rec_pb = protobuf.parse_message_from_gzipped_file(sensor_pb2.Recording, file_path)
            recordings.append(Recording.from_pb(rec_pb))
        com_rec = CombinedRecording.from_recordings(recordings)
        return com_rec

    except Exception:
        logging.exception('Could not read sensor data')
        return None


class TestSpeedingModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        protobuf_dir = '2361bd57947bca0f/2020-01-01/sensor/'
        protobuf_files = ['9f1b46574527e81d375d72c3cdc000d25ff91170', '554f4bec36b48546395f9631f8b93abb4e9ec5fb',
                          "791330e5ab0c04c73f8af9751ec5e188017e9706"]
        sensor_pb_gzip = [os.path.join(Path(__file__).resolve().parent, protobuf_dir, pb) for pb in protobuf_files]
        cls.com_rec = get_combined_recording(sensor_pb_gzip)

    @patch('nauto_zoo.models.speeding.SpeedingModel._do_post')
    def test_success(self, test_patch):
        test_patch.return_value = {'response': [{'start_index': 0, 'stop_index': 1, 'severity': 'low'},
                                                {'start_index': 17, 'stop_index': 22, 'severity': 'high'}],
                                   'success': True}

        model_input = ModelInput()
        model_input.set('sensor', self.com_rec)
        config_provided_by_ucm = {
            "speed_th0": 5,
            "speed_th1": 12,
            "speed_th2": 20,
            "time_th": 1,
            "time_scale": 10 ** -9,
        }
        model = SpeedingModel(config_provided_by_ucm)
        model.set_logger(Mock())
        res = model.run(model_input)

        expected = [{'start_index': 0, 'stop_index': 1, 'severity': 'low'},
                    {'start_index': 17, 'stop_index': 22, 'severity': 'high'}]

        self.assertCountEqual(res.raw_output, expected)
        self.assertEqual(res.summary, 'TRUE')
        self.assertEqual(res.score, 1.)
        self.assertEqual(res.confidence, 100)

    @patch('nauto_zoo.models.speeding.SpeedingModel._do_post')
    def test_unsuccess(self, test_patch):
        test_patch.return_value = {'response': [{'start_index': 0, 'stop_index': 1, 'severity': 'low'},
                                                {'start_index': 17, 'stop_index': 22, 'severity': 'high'}],
                                   'success': False}

        model_input = ModelInput()
        model_input.set('sensor', self.com_rec)
        config_provided_by_ucm = {
            "speed_th0": 5,
            "speed_th1": 12,
            "speed_th2": 20,
            "time_th": 1,
            "time_scale": 10 ** -9,
        }
        model = SpeedingModel(config_provided_by_ucm)
        model.set_logger(Mock())
        res = model.run(model_input)

        self.assertFalse(res.raw_output)
        self.assertEqual(res.summary, 'FALSE')
        self.assertEqual(res.score, 0.)
        self.assertEqual(res.confidence, 0)

    @patch('nauto_zoo.models.speeding.SpeedingModel._do_post')
    def test_no_speeding(self, test_patch):
        test_patch.return_value = {'response': [],
                                   'success': True}

        model_input = ModelInput()
        model_input.set('sensor', self.com_rec)
        config_provided_by_ucm = {
            "speed_th0": 5,
            "speed_th1": 12,
            "speed_th2": 20,
            "time_th": 1,
            "time_scale": 10 ** -9,
        }
        model = SpeedingModel(config_provided_by_ucm)
        model.set_logger(Mock())
        res = model.run(model_input)

        self.assertFalse(res.raw_output)
        self.assertEqual(res.summary, 'FALSE')
        self.assertEqual(res.score, 1.)
        self.assertEqual(res.confidence, 100)

    # It is a real test without mocking the post-request.
    # It can be used only for local development and testing.
    # Keep it commented for circle-ci testing.
    # def test_real_request(self):
    #     import logging
    #     logging.basicConfig(level=logging.DEBUG)
    #
    #     model_input = ModelInput()
    #     model_input.set('sensor', self.com_rec)
    #     config_provided_by_ucm = {
    #         "speed_th0": 5,
    #         "speed_th1": 12,
    #         "speed_th2": 20,
    #         "time_th": 1,
    #         "time_scale": 10 ** -9,
    #     }
    #     model = SpeedingModel(config_provided_by_ucm)
    #
    #     model.set_logger(logging.getLogger())
    #
    #     res = model.run(model_input)
    #
    #     expected = [{'start_index': 0, 'stop_index': 1, 'severity': 'low'},
    #                 {'start_index': 17, 'stop_index': 22, 'severity': 'high'}]
    #
    #     self.assertCountEqual(res.raw_output, expected)


if __name__ == '__main__':
    unittest.main()
