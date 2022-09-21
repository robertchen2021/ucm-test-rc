from nauto_datasets.utils import protobuf
from sensor import sensor_pb2
from nauto_datasets.core import sensors
from tests.sensor.test_sensor_proto import SENSOR_DATA_DIR
import unittest


class MCODBoundingBoxTest(unittest.TestCase):

    def test_existence(self):
        r_pb = protobuf.parse_message_from_gzipped_file(
            sensor_pb2.Recording,
            SENSOR_DATA_DIR / 'drowsiness/positive_case'
        )
        recording = sensors.Recording.from_pb(r_pb)

        self.assertTrue('system_ms' in recording.mcod_bounding_boxes._fields)
        self.assertTrue('sensor_ns' in recording.mcod_bounding_boxes._fields)
        self.assertTrue('bounding_box' in recording.mcod_bounding_boxes._fields)
        self.assertTrue('model_id' in recording.mcod_bounding_boxes._fields)
