"""
Run from console:
$ cd nauto-ai/nauto-ds-utils
$ python -m pytest tests/test_algo_imu.py --log-cli-level=INFO
"""

import os
import unittest
from pathlib import Path
from typing import List, Optional
import logging
import sys

import scipy as sp
from scipy.signal import butter

from nauto_datasets.core.sensors import CombinedRecording, Recording
from nauto_datasets.utils import protobuf
from nauto_ds_utils.algos_imu import startle_braking_candidate
from nauto_ds_utils.utils.sensor import get_combined_recording
from nauto_ds_utils.utils.sensor_filters import butter_lowpass
from sensor import sensor_pb2


class TestStartleBrakingCandidate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.b, cls.a = butter_lowpass(
            cutOff=5,
            fs=200,
            order=4)

        cls.a_mid_th = -3
        cls.buffet_length = 200
        cls.a_max_th = -6.25
        cls.slope_min = 3.25

    def test_true_braking_candidate(self):
        protobuf_dir = 'data/16513c82895dad08'
        protobuf_files = ['sensor_0.gz', 'sensor_1.gz',
                          'sensor_2.gz', 'sensor_3.gz']
        sensor_pb_gzip = [os.path.join(Path(__file__).resolve().parent,
                                       protobuf_dir,
                                       pb)
                          for pb in protobuf_files]
        com_rec = get_combined_recording(sensor_pb_gzip)
        acc = com_rec.oriented_acc.stream._to_df().sort_values(
            'sensor_ns').reset_index(drop=True)
        time_sec = (acc["sensor_ns"].values - acc["sensor_ns"].values[0]) / 1e9
        accx_lpf = sp.signal.filtfilt(self.b, self.a, acc["x"])

        response = startle_braking_candidate(time_sec,
                                             accx_lpf,
                                             self.slope_min,
                                             self.a_max_th, self.a_mid_th,
                                             self.buffet_length)

        self.assertEqual(len(response), 2)

    def test_false_braking_candidate(self):
        protobuf_dir = 'data/164d3017f233e587'
        protobuf_files = ['sensor_0.gz', 'sensor_1.gz', 'sensor_2.gz']
        sensor_pb_gzip = [os.path.join(Path(__file__).resolve().parent,
                                       protobuf_dir,
                                       pb)
                          for pb in protobuf_files]
        com_rec = get_combined_recording(sensor_pb_gzip)
        acc = com_rec.oriented_acc.stream._to_df().sort_values(
            'sensor_ns').reset_index(drop=True)
        time_sec = (acc["sensor_ns"].values - acc["sensor_ns"].values[0]) / 1e9
        accx_lpf = sp.signal.filtfilt(self.b, self.a, acc["x"])

        response = startle_braking_candidate(time_sec,
                                             accx_lpf, self.slope_min,
                                             self.a_max_th,
                                             self.a_mid_th,
                                             self.buffet_length)

        self.assertFalse(response, msg=sensor_pb_gzip)


if __name__ == '__main__':
    unittest.main()
