import numpy

from nauto_datasets.utils import protobuf
from sensor import sensor_pb2
from nauto_datasets.core import sensors
from tests.sensor.test_sensor_proto import SENSOR_DATA_DIR
import tensorflow as tf


class FCWRecordingTest(tf.test.TestCase):
    def test_enum_in_stream(self):
        r_pb = protobuf.parse_message_from_gzipped_file(
            sensor_pb2.Recording, SENSOR_DATA_DIR / 'fcw.pb.gz'
        )
        recording = sensors.Recording.from_pb(r_pb)
        speed_source = recording.speed.speed_source
        ones = numpy.empty(9, dtype=int)
        ones.fill(1)
        assert numpy.array_equal(ones, speed_source)

    def test_fcw(self):
        r_pb = protobuf.parse_message_from_gzipped_file(
            sensor_pb2.Recording, SENSOR_DATA_DIR / 'fcw.pb.gz'
        )
        recording = sensors.Recording.from_pb(r_pb)
        self.assertAllClose(recording.fcw.ttc, r_pb.fcw.ttc)
        self.assertAllClose(recording.fcw.distance_estimate, r_pb.fcw.distance_estimate)

        self.assertAllClose(recording.risk_assessment.should_play_rta, r_pb.risk_assessment.should_play_rta)
        self.assertAllClose(recording.risk_rta.rta_requested_sensor_ns, r_pb.risk_rta.rta_requested_sensor_ns)
