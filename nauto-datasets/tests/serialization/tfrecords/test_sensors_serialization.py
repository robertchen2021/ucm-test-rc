from pathlib import Path

import numpy as np
import tensorflow as tf

from  nauto_datasets.serialization.tfrecords import decoding as tf_decoding
import nauto_datasets.serialization.tfrecords.sensors as sensors_ser
from nauto_datasets.core import sensors
from nauto_datasets.core.streams import CombinedStreamMixin
from nauto_datasets.utils import protobuf
from sensor import sensor_pb2

SENSOR_DATA_DIR = Path(__file__).parents[2] / 'test_data' / 'sensor_data'


def get_recordings() -> sensors.CombinedRecording:
    for dir_name in [
            '2_10_1_quaternion_w',
            '2_11_2_imu_statistics',
            '2_10_distraction_multilabel',
            '2_10_1_oriented_data',
            '2_11_bboxes_tailgating'
        ] :
        files = list((SENSOR_DATA_DIR / dir_name).iterdir())
        files = sorted(files, key=lambda p: p.name)

        proto_msgs = [
            protobuf.parse_message_from_gzipped_file(
                sensor_pb2.Recording, file_path)
            for file_path in files
        ]

        com_recordings = sensors.CombinedRecording.from_recordings(
            [sensors.Recording.from_pb(r_pb)
                for r_pb in proto_msgs])
        yield com_recordings


def evaluate(tensors_structure):
    if isinstance(tensors_structure, tf.Tensor):
        return tensors_structure.numpy()
    elif isinstance(tensors_structure, tf.sparse.SparseTensor):
        return tf.sparse.to_dense(tensors_structure).numpy()
    else:
        return tf.nest.pack_sequence_as(
            tensors_structure,
            list(map(lambda t: t.numpy() if isinstance(t, tf.Tensor) else tf.sparse.to_dense(t).numpy(),
                     tf.nest.flatten(tensors_structure))))


class TestEncoding(tf.test.TestCase):

    def test_combined_recording_serialization(self):

        def compare_features(values, feature):
            if type(values) == float:
                self.assertAllClose(
                    feature.float_list.value,
                    [values])
            elif type(values) == int:
                self.assertAllEqual(
                    feature.int64_list.value,
                    [values])
            elif type(values) == bytes:
                self.assertAllEqual(
                    feature.bytes_list.value,
                    [values])
            elif type(values) == str:
                self.assertAllEqual(
                    feature.bytes_list.value,
                    [values.encode()])
            elif isinstance(values, np.ndarray):
                array = values
                if array.dtype.kind in ('i', 'u', 'b'):
                    self.assertAllEqual(
                        feature.int64_list.value,
                        array)
                elif array.dtype.kind == 'f':
                    self.assertAllClose(
                        feature.float_list.value,
                        array)
                elif array.dtype.kind in ('S', 'a'):
                    self.assertAllEqual(
                        feature.bytes_list.value,
                        array)
                elif array.dtype.kind == 'U':
                    self.assertAllEqual(
                        np.array(feature.bytes_list.value, dtype=np.str),
                        array)
                else:
                    raise ValueError(f'unknown kind {array.dtype.kind}')

        for com_recordings in get_recordings():
            features = sensors_ser.combined_recording_to_features(
                com_recordings)

            rec_dict = com_recordings._asdict()

            for key, val in rec_dict.items():
                # combined stream should be part of the context
                if issubclass(type(val), CombinedStreamMixin):
                    # compare lenghts of combined streams
                    lengths = features.context[f'{key}/lengths']
                    self.assertAllEqual(val.lengths, lengths.int64_list.value)
                    # compare values
                    for field_name, values in val.stream._asdict().items():
                        # e.g. BoundingBoxStream contains a list of BoundingBox
                        if isinstance(values, list):
                            if len(values) > 0:
                                for sub_field_name in values[0]._fields:
                                    self.assertEqual(
                                        len(features.feature_lists[f'{key}/stream/{field_name}/{sub_field_name}']),
                                        len(values))
                                    for ind, values_tuple in enumerate(values):
                                        compare_features(
                                            getattr(values_tuple, sub_field_name),
                                            features.feature_lists[f'{key}/stream/{field_name}/{sub_field_name}'][ind])
                        else:
                            feature = features.context[f'{key}/stream/{field_name}']
                            compare_features(values, feature)
                # lists should be part of feature lists
                elif isinstance(val, list):
                    if len(val) > 0:
                        for field_name in val[0]._fields:
                            self.assertEqual(
                                len(features.feature_lists[f'{key}/{field_name}']),
                                len(val))
                            for ind, values_tuple in enumerate(val):
                                compare_features(
                                    getattr(values_tuple, field_name),
                                    features.feature_lists[f'{key}/{field_name}'][ind])
                else:
                    raise ValueError(f'Unrecognized type {type(val)}')


class TestDecoding(tf.test.TestCase):

    def test_combined_recording_parsers_regular(self):
        for com_recording in get_recordings():

            features = sensors_ser.combined_recording_to_features(
                com_recording,
                ignore_sequence_features=False,
                use_tensor_protos=False)
            parsers = sensors_ser.combined_recording_parsers(
                ignore_sequence_features=False,
                parse_tensor_protos=False)

            seq_example = features.to_sequence_example()
            serialized_seq_example = seq_example.SerializeToString()

            seq_example_tensor = tf.constant(serialized_seq_example)
            context_tensors, sequence_tensors = parsers.parse_sequence_example(seq_example_tensor)

            [context_vals, sequence_vals] = evaluate([context_tensors, sequence_tensors])

            # some equality checks
            self.assertAllClose(
                context_vals['acc/stream/z'],
                com_recording.acc.stream.z)
            self.assertAllEqual(
                context_vals['acc/lengths'],
                com_recording.acc.lengths)

            self.assertAllClose(
                context_vals['gyro/stream/y'],
                com_recording.gyro.stream.y)
            self.assertAllEqual(
                context_vals['gyro/stream/sensor_ns'],
                com_recording.gyro.stream.sensor_ns)

            self.assertAllClose(
                context_vals['mag/stream/x'],
                com_recording.mag.stream.x)
            self.assertAllEqual(
                context_vals['mag/stream/system_ms'],
                com_recording.mag.stream.system_ms)

            self.assertAllClose(
                context_vals['ekf/stream/speed'],
                com_recording.ekf.stream.speed)
            self.assertAllClose(
                context_vals['ekf/stream/acc_y'],
                com_recording.ekf.stream.acc_y)

            self.assertAllClose(
                context_vals['dist/stream/score'],
                com_recording.dist.stream.score)
            self.assertAllEqual(
                context_vals['dist/stream/sensor_ns'],
                com_recording.dist.stream.sensor_ns)

            self.assertAllClose(
                context_vals['tailgating/stream/score'],
                com_recording.tailgating.stream.score)
            self.assertAllEqual(
                context_vals['tailgating/stream/sensor_ns'],
                com_recording.tailgating.stream.sensor_ns)
            self.assertAllEqual(
                context_vals['tailgating/stream/distance_estimate'],
                com_recording.tailgating.stream.distance_estimate)

            self.assertAllClose(
                context_vals['imu_statistics/stream/ax_first_moment'],
                com_recording.imu_statistics.stream.ax_first_moment)
            self.assertAllClose(
                context_vals['imu_statistics/stream/gx_second_moment'],
                com_recording.imu_statistics.stream.gx_second_moment)
            self.assertAllEqual(
                context_vals['imu_statistics/stream/accel_sensor_ns'],
                com_recording.imu_statistics.stream.accel_sensor_ns)

            self.assertAllEqual(
                context_vals['bounding_boxes_external/stream/sensor_ns'],
                com_recording.bounding_boxes_external.stream.sensor_ns)

            # sparse tensors for sequence values
            tops = sequence_vals['bounding_boxes_external/stream/bounding_box/top']
            for ind, bbs in enumerate(
                    com_recording.bounding_boxes_external.stream.bounding_box):
                self.assertAllClose(tops[ind][:len(bbs.top)], bbs.top)

            obj_types = sequence_vals['bounding_boxes_external/stream/bounding_box/objectType']
            for ind, bbs in enumerate(
                    com_recording.bounding_boxes_external.stream.bounding_box):
                self.assertAllClose(obj_types[ind][:len(bbs.objectType)], bbs.objectType)

            # dense tensors for sequence values
            self.assertAllClose(
                sequence_vals['ekf_configs/rot_angle_x'],
                [config.rot_angle_x for config in com_recording.ekf_configs])
            self.assertAllEqual(
                sequence_vals['ekf_configs/rot_count'],
                [config.rot_count for config in com_recording.ekf_configs])
            self.assertAllClose(
                sequence_vals['ekf_configs/sigma_ax'],
                [config.sigma_ax for config in com_recording.ekf_configs])

            self.assertAllEqual(
                sequence_vals['model_definition/name'],
                [m_def.name for m_def in com_recording.model_definition])
            self.assertAllEqual(
                sequence_vals['model_definition/version'],
                [m_def.version for m_def in com_recording.model_definition])

            self.assertAllEqual(
                sequence_vals['metadatas/version'],
                [meta.version.encode() for meta in com_recording.metadatas])
            self.assertAllEqual(
                sequence_vals['metadatas/utc_boot_time_offset_ns'],
                [meta.utc_boot_time_offset_ns for meta in com_recording.metadatas])

            # sparse tensors for sequence values
            configs_x = sequence_vals['ekf_configs/config_x']

            self.assertAllClose(
                configs_x,
                [config.config_x for config in com_recording.ekf_configs])

    def test_combined_recording_parsers_protos(self):
        for com_recording in get_recordings():
            features = sensors_ser.combined_recording_to_features(
                com_recording,
                ignore_sequence_features=True,
                use_tensor_protos=True)
            parsers = sensors_ser.combined_recording_parsers(
                ignore_sequence_features=True,
                parse_tensor_protos=True)

            example = features.to_example()
            serialized_example = example.SerializeToString()

            example_tensor = tf.constant(serialized_example)
            feature_tensors = parsers.parse_example(example_tensor)

            context_vals = evaluate(feature_tensors)

            # some equality checks
            self.assertAllClose(
                context_vals['acc/stream/x'],
                com_recording.acc.stream.x)
            self.assertAllEqual(
                context_vals['acc/lengths'],
                com_recording.acc.lengths)

            self.assertAllClose(
                context_vals['gyro/stream/y'],
                com_recording.gyro.stream.y)
            self.assertAllEqual(
                context_vals['gyro/stream/sensor_ns'],
                com_recording.gyro.stream.sensor_ns)

            self.assertAllClose(
                context_vals['mag/stream/z'],
                com_recording.mag.stream.z)
            self.assertAllEqual(
                context_vals['mag/stream/system_ms'],
                com_recording.mag.stream.system_ms)

            self.assertAllClose(
                context_vals['ekf/stream/speed'],
                com_recording.ekf.stream.speed)
            self.assertAllClose(
                context_vals['ekf/stream/acc_y'],
                com_recording.ekf.stream.acc_y)

            self.assertAllClose(
                context_vals['dist/stream/score'],
                com_recording.dist.stream.score)
            self.assertAllEqual(
                context_vals['dist/stream/sensor_ns'],
                com_recording.dist.stream.sensor_ns)

            self.assertAllClose(
                context_vals['loose_device/stream/heuristic_score'],
                com_recording.loose_device.stream.heuristic_score)

            self.assertAllEqual(
                context_vals['device_orientation/stream/converged'],
                com_recording.device_orientation.stream.converged)

            self.assertAllEqual(
                context_vals['dist_multilabel/stream/score_no_face'],
                com_recording.dist_multilabel.stream.score_no_face)
