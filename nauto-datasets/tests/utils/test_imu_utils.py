import numpy as np
import tensorflow as tf
from typing import NamedTuple
from pathlib import Path
from typing import List
from nauto_datasets.core.sensors import CombinedRecording, Recording
from nauto_datasets.utils import protobuf
from sensor import sensor_pb2
from nauto_datasets.imu_utils import convert_oriented_to_raw


class TestIMUUtils(tf.test.TestCase):
    def test_imu_rotation(self):
        sensor_files = [
            str(Path('./tests/test_data/sensor_data/rotation_test/a9f9f4cdce5663fe2f56481775a4d195657de60a').resolve())]

        sensor_data = [open(file_name, "rb").read() for file_name in sensor_files]
        recording = CombinedRecording.from_recordings(
            [Recording.from_pb(protobuf.parse_message_from_gzipped_bytes(sensor_pb2.Recording, recording_bytes)) for
             recording_bytes in sensor_data])

        computed_raw_acc, computed_raw_gyro = convert_oriented_to_raw(recording)

        assert computed_raw_acc.stream.x.shape[0] == 1994
        assert abs(round(computed_raw_acc.stream.x[0], 3) - 4.452) < 0.001

