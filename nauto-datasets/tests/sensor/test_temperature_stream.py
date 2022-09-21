from nauto_datasets.utils import protobuf
from sensor import sensor_pb2
from nauto_datasets.core import sensors
from tests.sensor.test_sensor_proto import SENSOR_DATA_DIR
import unittest


class TemperatureTest(unittest.TestCase):

    def test_existence(self):
        r_pb = protobuf.parse_message_from_gzipped_file(
            sensor_pb2.Recording,
            SENSOR_DATA_DIR / 'drowsiness/positive_case'
        )
        recording = sensors.Recording.from_pb(r_pb)

        self.assertTrue('temperature' in recording._fields)
        self.assertTrue('system_ms' in recording.temperature._fields)
        self.assertTrue(
            'cpu_temperature_celsius' in recording.temperature._fields)
        self.assertTrue(
            'gpu_temperature_celsius' in recording.temperature._fields)
        self.assertTrue(
            'computation_capacity' in recording.temperature._fields)
        self.assertTrue('sensor_ns' in recording.temperature._fields)
