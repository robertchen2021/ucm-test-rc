import unittest
from pathlib import Path

from sensor import sensor_pb2
from nauto_datasets.utils import protobuf


SENSOR_DATA_DIR = Path(__file__).parents[1] / 'test_data' / 'sensor_data'


class TestSensorDeserialization(unittest.TestCase):

    def test_recordings_from_file(self):
        gzipped_files = SENSOR_DATA_DIR.glob('*.pb')
        for file_path in gzipped_files:
            msg = protobuf.parse_message_from_file(
                sensor_pb2.Recording,
                file_path)

            self.assertIsInstance(msg, sensor_pb2.Recording)

    def test_recordings_from_gzipped_file(self):
        gzipped_files = SENSOR_DATA_DIR.glob('*.pb.gz')
        for file_path in gzipped_files:
            msg = protobuf.parse_message_from_gzipped_file(
                sensor_pb2.Recording,
                file_path)

            self.assertIsInstance(msg, sensor_pb2.Recording)
