from nauto_datasets.utils import protobuf
from sensor import sensor_pb2
from nauto_datasets.core import sensors
from tests.sensor.test_sensor_proto import SENSOR_DATA_DIR
import unittest


class OmniFusionLocationTest(unittest.TestCase):

    def test_existence(self):
        r_pb = protobuf.parse_message_from_gzipped_file(
            sensor_pb2.Recording,
            SENSOR_DATA_DIR / 'of_data/ofvd.gz'
        )
        recording = sensors.Recording.from_pb(r_pb)

        self.assertTrue(
            'batch_sensor_ns' in recording.omnifusion_location._fields)
        self.assertTrue(
            'batch_system_ms' in recording.omnifusion_location._fields)
        self.assertTrue(
            'sample_done_sensor_ns' in recording.omnifusion_location._fields)
        self.assertTrue('sensor_ns' in recording.omnifusion_location._fields)
        self.assertTrue('initialized' in recording.omnifusion_location._fields)
        self.assertTrue('report_tag' in recording.omnifusion_location._fields)
        self.assertTrue('latitude_deg' in recording.omnifusion_location._fields)
        self.assertTrue('latitude_accuracy_deg' in recording.omnifusion_location._fields)
        self.assertTrue('longitude_deg' in recording.omnifusion_location._fields)
        self.assertTrue(
            'longitude_accuracy_deg' in recording.omnifusion_location._fields)
        self.assertTrue('altitude_m' in recording.omnifusion_location._fields)
        self.assertTrue('altitude_accuracy_m' in recording.omnifusion_location._fields)
        self.assertTrue('speed_mps' in recording.omnifusion_location._fields)
        self.assertTrue('speed_accuracy_mps' in recording.omnifusion_location._fields)
        self.assertTrue('bearing_deg' in recording.omnifusion_location._fields)
        self.assertTrue('bearing_accuracy_deg' in recording.omnifusion_location._fields)
        self.assertTrue(
            'longitudinal_velocity_mps' in recording.omnifusion_location._fields)
        self.assertTrue(
            'longitudinal_velocity_accuracy_mps' in recording.omnifusion_location._fields)
        self.assertTrue('north_velocity_mps' in recording.omnifusion_location._fields)
        self.assertTrue(
            'north_velocity_accuracy_mps' in recording.omnifusion_location._fields)
        self.assertTrue('east_velocity_mps' in recording.omnifusion_location._fields)
        self.assertTrue(
            'east_velocity_accuracy_mps' in recording.omnifusion_location._fields)
        self.assertTrue(
            'down_velocity_accuracy_mps' in recording.omnifusion_location._fields)
        self.assertTrue('down_velocity_mps' in recording.omnifusion_location._fields)