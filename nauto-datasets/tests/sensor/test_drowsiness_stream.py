from nauto_datasets.utils import protobuf
from sensor import sensor_pb2
from nauto_datasets.core import sensors
from tests.sensor.test_sensor_proto import SENSOR_DATA_DIR
import unittest


class DrowsinessTest(unittest.TestCase):


    def test_negative_case(self):
        r_pb = protobuf.parse_message_from_gzipped_file(
            sensor_pb2.Recording,
            SENSOR_DATA_DIR / 'drowsiness/negative_case'
        )
        recording = sensors.Recording.from_pb(r_pb)
        # assert if the field is in there
        self.assertTrue('drowsiness' in recording._fields)
        self.assertTrue(all(recording.drowsiness.score == [
            0.053955078125, 0.048553466796875, 0.050994873046875]))
        self.assertTrue(
            all(recording.drowsiness.isDrowsy == [False, False, False]))

    def test_positive_case(self):
        r_pb = protobuf.parse_message_from_gzipped_file(
            sensor_pb2.Recording,
            SENSOR_DATA_DIR / 'drowsiness/positive_case'
        )
        recording = sensors.Recording.from_pb(r_pb)
        self.assertTrue('drowsiness' in recording._fields)
        self.assertTrue(all(recording.drowsiness.score ==
                            [0.54296875, 0.60791015625, 0.70849609375,
                             0.0181121826171875, 0.0183868408203125, 0.0190887451171875,
                             0.019378662109375, 0.0195465087890625, 0.01983642578125]))
        self.assertTrue(all(recording.drowsiness.isDrowsy ==
                            [False, False, True,
                             False, False, False,
                             False, False, False]))
