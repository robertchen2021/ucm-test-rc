from nauto_datasets.utils import protobuf
from sensor import sensor_pb2
from nauto_datasets.core import sensors
from tests.sensor.test_sensor_proto import SENSOR_DATA_DIR
import unittest


class OmniFusionVehicleDynamicsTest(unittest.TestCase):

    def test_existence(self):
        r_pb = protobuf.parse_message_from_gzipped_file(
            sensor_pb2.Recording,
            SENSOR_DATA_DIR / 'of_data/ofvd.gz'
        )
        recording = sensors.Recording.from_pb(r_pb)

        self.assertTrue(
            'batch_sensor_ns' in recording.omnifusion_vehicle_dynamics._fields)
        self.assertTrue(
            'batch_system_ms' in recording.omnifusion_vehicle_dynamics._fields)
        self.assertTrue(
            'sample_done_sensor_ns' in recording.omnifusion_vehicle_dynamics._fields)
        self.assertTrue('sensor_ns' in recording.omnifusion_vehicle_dynamics._fields)
        self.assertTrue('initialized' in recording.omnifusion_vehicle_dynamics._fields)
        self.assertTrue('report_tag' in recording.omnifusion_vehicle_dynamics._fields)
        self.assertTrue('xfilt_accel' in recording.omnifusion_vehicle_dynamics._fields)
        self.assertTrue('xfilt_brake' in recording.omnifusion_vehicle_dynamics._fields)
        self.assertTrue('xfilt_lcorn' in recording.omnifusion_vehicle_dynamics._fields)
        self.assertTrue('xfilt_rcorn' in recording.omnifusion_vehicle_dynamics._fields)
