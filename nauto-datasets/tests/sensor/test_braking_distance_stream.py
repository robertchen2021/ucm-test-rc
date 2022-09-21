from nauto_datasets.utils import protobuf
from sensor import sensor_pb2
from nauto_datasets.core import sensors
from tests.sensor.test_sensor_proto import SENSOR_DATA_DIR
import unittest


class BrakingDistanceTest(unittest.TestCase):

    def test_existence(self):
        r_pb = protobuf.parse_message_from_gzipped_file(
            sensor_pb2.Recording,
            SENSOR_DATA_DIR / 'drowsiness/positive_case'
        )
        recording = sensors.Recording.from_pb(r_pb)

        self.assertTrue('system_ms' in recording.braking_distance._fields)
        self.assertTrue('speed_sensor_ns' in recording.braking_distance._fields)
        self.assertTrue('acc_sensor_ns' in recording.braking_distance._fields)
        self.assertTrue(
            'predicted_braking_distance_m' in recording.braking_distance._fields)
        self.assertTrue(
            'predicted_distracted_braking_distance_m' in recording.braking_distance._fields)
