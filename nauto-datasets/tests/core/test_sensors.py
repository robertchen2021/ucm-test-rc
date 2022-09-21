import inspect
from pathlib import Path

import numpy as np
import tensorflow as tf

from nauto_datasets.core import sensors, streams
from nauto_datasets.utils import protobuf
from nauto_datasets.utils import transformations as nauto_trans
from sensor import sensor_pb2

SENSOR_DATA_DIR = Path(__file__).parents[1] / "test_data" / "sensor_data"


class RecordingTest(tf.test.TestCase):
    def test_reading_from_pb(self):
        paths = [
            SENSOR_DATA_DIR
            / "2_10_1_quaternion_w"
            / "00916-imu-20181022-220133-0700.pb.gz",
            SENSOR_DATA_DIR
            / "2_10_distraction_multilabel"
            / "01113-imu-20180920-114611-0700.pb.gz",
            SENSOR_DATA_DIR
            / "2_10_loosedevice"
            / "9805873133f69be3527d4282110481ce6ac27985.pb.gz",
            SENSOR_DATA_DIR
            / "2_11_2_imu_statistics"
            / "2a4dcb0a0519e9a7-2019-03-27-sensor-2774c23b3d016d3209a58cf83586702440c39d7e.gz",
            SENSOR_DATA_DIR
            / "2_11_bboxes_tailgating"
            / "00039-imu-20190212-104624-0800.pb.gz",
            SENSOR_DATA_DIR
            / "2_12_distraction_multilabel"
            / "03808-imu-20190429-134136-0700.pb.gz",
            SENSOR_DATA_DIR
            / "2_12_tailgating"
            / "03787-imu-20190429-133803-0700.pb.gz",
        ]
        for file_path in paths:
            r_pb = protobuf.parse_message_from_gzipped_file(
                sensor_pb2.Recording, file_path
            )

            recording = sensors.Recording.from_pb(r_pb)

            # some matadata checks
            self.assertEqual(recording.metadata.version, r_pb.version)
            self.assertEqual(
                recording.metadata.capture_start_system_ms, r_pb.capture_start_system_ms
            )
            self.assertEqual(recording.metadata.boot_session, r_pb.boot_session)

            # some model_definition checks
            self.assertEqual(
                len(recording.model_definition), len(list(r_pb.model_definition))
            )
            for ind, m_def in enumerate(recording.model_definition):
                self.assertEqual(m_def.name, r_pb.model_definition[ind].name)
                self.assertEqual(m_def.version, r_pb.model_definition[ind].version)

            # some acc checks
            self.assertEqual(recording.acc._size(), len(r_pb.acc.x))
            self.assertAllClose(
                recording.acc.y,
                nauto_trans.delta_decompress(np.array(r_pb.acc.y), r_pb.acc.scale),
            )

            # some gyro checks
            self.assertEqual(recording.gyro._size(), len(r_pb.gyro.x))
            self.assertAllClose(
                recording.gyro.z,
                nauto_trans.delta_decompress(np.array(r_pb.gyro.z), r_pb.gyro.scale),
            )

            # some grv checks
            self.assertEqual(recording.grv._size(), len(r_pb.grv.x))
            self.assertAllClose(
                recording.grv.x,
                nauto_trans.delta_decompress(np.array(r_pb.grv.x), r_pb.grv.scale),
            )

            # grv - quaternion w added - not available everywhere
            if len(r_pb.grv.w) > 0:
                self.assertEqual(recording.grv._size(), len(r_pb.grv.w))
                self.assertAllClose(
                    recording.grv.w,
                    nauto_trans.delta_decompress(np.array(r_pb.grv.w), r_pb.grv.scale),
                )

            # some mag checks
            self.assertEqual(recording.mag._size(), len(r_pb.mag.x))
            self.assertAllClose(
                recording.mag.y,
                nauto_trans.delta_decompress(np.array(r_pb.mag.y), r_pb.mag.scale),
            )

            # lin check (must be empty)
            self.assertEqual(recording.lin._size(), len(r_pb.lin.x))
            self.assertAllClose(
                recording.lin.z,
                nauto_trans.delta_decompress(np.array(r_pb.lin.z), r_pb.lin.scale),
            )

            # some dist checks
            self.assertEqual(recording.dist._size(), len(r_pb.dist.score))
            self.assertAllClose(recording.dist.score, r_pb.dist.score)

            # some tails checks
            self.assertEqual(recording.tail._size(), len(r_pb.tail.score))
            self.assertAllClose(recording.tail.score, r_pb.tail.score)

            # some aqbit checks
            self.assertAllClose(recording.aqbit.gyro_z, r_pb.aqbit.gyro_z)
            self.assertAllEqual(recording.aqbit.sensor_ns, r_pb.aqbit.sensor_ns)
            self.assertAllClose(recording.aqbit.heading, r_pb.aqbit.heading)

            # some gps checks
            self.assertAllClose(recording.gps.speed, r_pb.gps.speed)

            # some ekf checks
            self.assertEqual(recording.ekf._size(), len(r_pb.ekf.acc_x))
            self.assertAllClose(recording.ekf.rsv1, r_pb.ekf.rsv1)

            # some obd checks
            self.assertEqual(recording.obd._size(), len(r_pb.obd.sensor_ns))
            self.assertAllEqual(
                recording.obd.brick_code, np.array(r_pb.obd.brick_code, dtype=np.str)
            )
            self.assertAllClose(recording.obd.value, r_pb.obd.value)

            # some device orientation checks
            self.assertEqual(
                recording.device_orientation._size(),
                len(r_pb.device_orientation.sensor_ns),
            )
            self.assertAllEqual(
                recording.device_orientation.name,
                np.array(r_pb.device_orientation.name, dtype=np.str),
            )
            self.assertAllClose(
                recording.device_orientation.pitch, recording.device_orientation.pitch
            )

            # some distraction multilabel checks
            self.assertEqual(
                recording.dist_multilabel._size(), len(r_pb.dist_multilabel.sensor_ns)
            )
            self.assertAllClose(
                recording.dist_multilabel.score_cell_phone,
                np.array(r_pb.dist_multilabel.score_cell_phone, dtype=np.float),
            )

            # some loose device checks
            self.assertEqual(
                recording.loose_device._size(), len(r_pb.loose_device.sensor_ns)
            )
            self.assertAllClose(
                recording.loose_device.heuristic_score,
                np.array(r_pb.loose_device.heuristic_score, dtype=np.float),
            )

            # some applied orientation checks
            self.assertEqual(
                recording.applied_orientation._size(),
                len(r_pb.applied_orientation.sensor_ns),
            )
            self.assertAllClose(
                recording.applied_orientation.yaw,
                np.array(r_pb.applied_orientation.yaw, dtype=np.float),
            )

            # some oriented acc stream checks
            self.assertEqual(recording.oriented_acc._size(), len(r_pb.oriented_acc.y))
            self.assertAllClose(
                recording.oriented_acc.z,
                nauto_trans.delta_decompress(
                    np.array(r_pb.oriented_acc.z), r_pb.oriented_acc.scale
                ),
            )

            # some oriented gyro stream checks
            self.assertEqual(recording.oriented_gyro._size(), len(r_pb.oriented_gyro.y))
            self.assertAllClose(
                recording.oriented_gyro.z,
                nauto_trans.delta_decompress(
                    np.array(r_pb.oriented_gyro.z), r_pb.oriented_gyro.scale
                ),
            )

    def test_reading_from_pb_distraction(self):
        for file_path in [
            SENSOR_DATA_DIR
            / "2_10_distraction_multilabel"
            / "01108-imu-20180920-114521-0700.pb.gz",
            SENSOR_DATA_DIR
            / "2_12_distraction_multilabel"
            / "03805-imu-20190429-134103-0700.pb.gz",
        ]:
            r_pb = protobuf.parse_message_from_gzipped_file(
                sensor_pb2.Recording, file_path
            )
            recording = sensors.Recording.from_pb(r_pb)
            for name in sensors.DistractionStream._fields:
                self.assertAllClose(
                    getattr(recording.dist_multilabel, name),
                    np.array(getattr(r_pb.dist_multilabel, name), dtype=np.float),
                )

    def test_reading_from_pb_tailgating(self):
        for file_path in [
            SENSOR_DATA_DIR
            / "2_12_tailgating"
            / "03789-imu-20190429-133823-0700.pb.gz",
            SENSOR_DATA_DIR
            / "2_12_tailgating"
            / "03794-imu-20190429-133913-0700.pb.gz",
        ]:
            r_pb = protobuf.parse_message_from_gzipped_file(
                sensor_pb2.Recording, file_path
            )
            recording = sensors.Recording.from_pb(r_pb)
            for name in sensors.TailgatingStream._fields:
                self.assertAllClose(
                    getattr(recording.tailgating, name),
                    np.array(getattr(r_pb.tailgating, name), dtype=np.float),
                )

    def test_reading_from_pb_loose_device(self):
        file_path = (
            SENSOR_DATA_DIR
            / "2_10_loosedevice"
            / "9192143624dc03451ca727c1f56f917684c5baad.pb.gz"
        )
        r_pb = protobuf.parse_message_from_gzipped_file(sensor_pb2.Recording, file_path)
        recording = sensors.Recording.from_pb(r_pb)
        for name in sensors.LooseDeviceStream._fields:
            self.assertAllClose(
                getattr(recording.loose_device, name),
                np.array(getattr(r_pb.loose_device, name), dtype=np.float),
            )

    def test_reading_from_pb_oriented_data(self):
        file_path = (
            SENSOR_DATA_DIR
            / "2_10_1_oriented_data"
            / "00434-imu-20181021-150051-0700.pb.gz"
        )
        r_pb = protobuf.parse_message_from_gzipped_file(sensor_pb2.Recording, file_path)
        recording = sensors.Recording.from_pb(r_pb)
        self.assertAllClose(
            recording.device_orientation.pitch,
            np.array(r_pb.device_orientation.pitch, dtype=np.float64),
        )
        self.assertAllEqual(
            recording.device_orientation.name, np.array(r_pb.device_orientation.name)
        )

        self.assertAllClose(
            recording.applied_orientation.roll,
            np.array(r_pb.applied_orientation.roll, dtype=np.float64),
        )

        self.assertAllEqual(
            recording.oriented_gyro.sensor_ns,
            nauto_trans.delta_decompress(np.array(r_pb.oriented_gyro.sensor_ns)),
        )

        self.assertAllClose(
            recording.oriented_gyro.w,
            nauto_trans.delta_decompress(
                np.array(r_pb.oriented_gyro.w), r_pb.oriented_gyro.scale
            ),
        )

        self.assertAllEqual(
            recording.oriented_acc.sensor_ns,
            nauto_trans.delta_decompress(np.array(r_pb.oriented_acc.sensor_ns)),
        )

        self.assertAllClose(
            recording.oriented_acc.x,
            nauto_trans.delta_decompress(
                np.array(r_pb.oriented_acc.x), r_pb.oriented_acc.scale
            ),
        )

    def test_reading_from_pb_bboxes_and_tailgating(self):
        file_path = (
            SENSOR_DATA_DIR
            / "2_11_bboxes_tailgating"
            / "00047-imu-20190212-104744-0800.pb.gz"
        )
        r_pb = protobuf.parse_message_from_gzipped_file(sensor_pb2.Recording, file_path)
        recording = sensors.Recording.from_pb(r_pb)

        self.assertAllClose(
            recording.bounding_boxes_external.sensor_ns,
            np.array(r_pb.bounding_boxes_external.sensor_ns),
        )
        self.assertAllEqual(
            recording.bounding_boxes_external.model_id,
            np.array(r_pb.bounding_boxes_external.model_id),
        )
        for i in range(len(r_pb.bounding_boxes_external.bounding_box)):
            self.assertAllEqual(
                recording.bounding_boxes_external.bounding_box[i].right,
                np.array(r_pb.bounding_boxes_external.bounding_box[i].right),
            )

        self.assertAllEqual(
            recording.tailgating.front_box_index,
            np.array(r_pb.tailgating.front_box_index),
        )

    def test_reading_from_pb_imu_statistics(self):
        file_path = (
            SENSOR_DATA_DIR
            / "2_11_2_imu_statistics"
            / "2a4dcb0a0519e9a7-2019-03-27-sensor-2664751bb390e957e055cf7708d69797c5adaa77.gz"
        )
        r_pb = protobuf.parse_message_from_gzipped_file(sensor_pb2.Recording, file_path)
        recording = sensors.Recording.from_pb(r_pb)

        for field_name, field_val in zip(
            recording.imu_statistics._fields, recording.imu_statistics
        ):
            self.assertAllClose(
                field_val, np.array(getattr(r_pb.imu_statistics, field_name))
            )

    def test_bboxes_model_mapping(self):
        file_paths = [
            SENSOR_DATA_DIR
            / "2_11_bboxes_tailgating"
            / "00047-imu-20190212-104744-0800.pb.gz",
            SENSOR_DATA_DIR
            / "2_12_tailgating"
            / "03793-imu-20190429-133903-0700.pb.gz",
            SENSOR_DATA_DIR
            / "2_12_distraction_multilabel"
            / "03801-imu-20190429-134023-0700.pb.gz",
        ]
        for file_path in file_paths:
            r_pb = protobuf.parse_message_from_gzipped_file(
                sensor_pb2.Recording, file_path
            )
            recording = sensors.Recording.from_pb(r_pb)
            self.assertAllEqual(
                recording.bounding_boxes_external.model_id,
                np.array(r_pb.bounding_boxes_external.model_id),
            )

            for m_id in recording.bounding_boxes_external.model_id:
                self.assertLess(m_id, len(r_pb.model_definition))

    def test_utc_time_conversion(self):
        file_path = (
            SENSOR_DATA_DIR
            / "2_11_2_imu_statistics"
            / "2a4dcb0a0519e9a7-2019-03-27-sensor-2664751bb390e957e055cf7708d69797c5adaa77.gz"
        )
        r_pb = protobuf.parse_message_from_gzipped_file(sensor_pb2.Recording, file_path)
        recording = sensors.Recording.from_pb(r_pb)

        utc_recording = recording.to_utc_time()
        for name, field_t in sensors.Recording._field_types.items():
            orig_stream = getattr(recording, name)
            new_stream = getattr(utc_recording, name)
            if inspect.isclass(field_t) and issubclass(field_t, sensors.SensorStream):
                ns_names = ["sensor_ns"]
            elif field_t == sensors.ImuStatisticsStream:
                ns_names = ["accel_sensor_ns", "gyro_sensor_ns"]

            if hasattr(field_t, "_field_types"):
                for sub_name, sub_field_t in field_t._field_types.items():
                    if sub_name in ns_names:
                        self.assertAllEqual(
                            getattr(new_stream, sub_name),
                            nauto_trans.to_utc_time(
                                getattr(orig_stream, sub_name),
                                recording.metadata.utc_boot_time_ns,
                                recording.metadata.utc_boot_time_offset_ns,
                            ),
                        )
                    else:
                        self.assertAllEqual(
                            getattr(orig_stream, sub_name),
                            getattr(new_stream, sub_name),
                        )


class BoundingBoxTest(tf.test.TestCase):
    def get_example(self) -> sensors.BoundingBox:
        count = 50
        return sensors.BoundingBox(
            left=np.arange(0, count, dtype=np.float32),
            top=np.arange(0, count, dtype=np.float32),
            right=np.arange(1, count + 1, dtype=np.float32),
            bottom=np.arange(0, count, dtype=np.float32) * 2,
            objectType=np.zeros([count], dtype=np.int32),
            score=np.random.uniform(0, 1.0, count),
        )

    def test_get_bbox_array(self):
        ex = self.get_example()
        arr = ex.get_bbox_array()
        self.assertAllEqual(arr.shape, [ex._size(), 4])
        self.assertAllClose(arr, np.c_[ex.left, ex.top, ex.right, ex.bottom])

    def test_get_bbox_sizes(self):
        ex = self.get_example()
        sizes = ex.get_bbox_sizes()
        self.assertAllEqual(sizes.shape, [ex._size(), 2])
        self.assertAllClose(sizes, np.c_[ex.right - ex.left, ex.bottom - ex.top])

    def test_get_bbox_areas(self):
        ex = self.get_example()
        areas = ex.get_bbox_areas()
        self.assertAllEqual(areas.shape, [ex._size()])
        self.assertAllClose(areas, (ex.right - ex.left) * (ex.bottom - ex.top))

    def test_to_normalized_coordinates(self):
        ex = self.get_example()
        img_height = 100
        img_width = 1000
        norm_ex = ex.to_normalized_coordinates(img_width, img_height)
        self.assertAllClose(
            norm_ex.get_bbox_array() * [img_width, img_height, img_width, img_height],
            ex.get_bbox_array(),
        )

    def test_to_absolute_coordinates(self):
        ex = self.get_example()
        img_width = 1000
        img_height = 100
        norm_ex = ex.to_normalized_coordinates(img_width, img_height)
        self.assertAllClose(
            norm_ex.to_absolute_coordinates(img_width, img_height).get_bbox_array(),
            ex.get_bbox_array(),
        )


class CombinedRecordingTest(tf.test.TestCase):
    def test_from_recordings(self):
        file_pairs = [
            (
                SENSOR_DATA_DIR
                / "2_11_2_imu_statistics"
                / "2a4dcb0a0519e9a7-2019-03-27-sensor-2774c23b3d016d3209a58cf83586702440c39d7e.gz",
                SENSOR_DATA_DIR
                / "2_11_2_imu_statistics"
                / "2a4dcb0a0519e9a7-2019-03-27-sensor-4b06fee6d55a3a781f26f420dadf70f4fc16cc70.gz",
            ),
            (
                SENSOR_DATA_DIR
                / "2_12_tailgating"
                / "03792-imu-20190429-133853-0700.pb.gz",
                SENSOR_DATA_DIR
                / "2_12_tailgating"
                / "03795-imu-20190429-133923-0700.pb.gz",
            ),
            (
                SENSOR_DATA_DIR
                / "2_12_distraction_multilabel"
                / "03799-imu-20190429-134003-0700.pb.gz",
                SENSOR_DATA_DIR
                / "2_12_distraction_multilabel"
                / "03805-imu-20190429-134103-0700.pb.gz",
            ),
        ]
        for file_path_1, file_path_2 in file_pairs:

            r_pb_1 = protobuf.parse_message_from_gzipped_file(
                sensor_pb2.Recording, file_path_1
            )
            r_pb_2 = protobuf.parse_message_from_gzipped_file(
                sensor_pb2.Recording, file_path_2
            )

            bb_s_size_1 = len(r_pb_1.bounding_boxes_external.bounding_box)
            bb_s_size_2 = len(r_pb_2.bounding_boxes_external.bounding_box)
            bb_s_counts_1 = [
                len(bb.left) for bb in r_pb_1.bounding_boxes_external.bounding_box
            ]
            bb_s_counts_2 = [
                len(bb.left) for bb in r_pb_2.bounding_boxes_external.bounding_box
            ]

            recording_1 = sensors.Recording.from_pb(r_pb_1)
            recording_2 = sensors.Recording.from_pb(r_pb_2)

            combined_recording = sensors.CombinedRecording.from_recordings(
                [recording_1, recording_2]
            )

            # XYZ streams
            for frame_name in [
                "acc",
                "gyro",
                "grv",
                "mag",
                "lin",
                "oriented_acc",
                "oriented_gyro",
            ]:
                self.assertAllEqual(
                    getattr(combined_recording, frame_name).stream.sensor_ns,
                    np.concatenate(
                        [
                            getattr(recording_1, frame_name).sensor_ns,
                            getattr(recording_2, frame_name).sensor_ns,
                        ]
                    ),
                )

                self.assertAllEqual(
                    getattr(combined_recording, frame_name).stream.system_ms,
                    np.concatenate(
                        [
                            getattr(recording_1, frame_name).system_ms,
                            getattr(recording_2, frame_name).system_ms,
                        ]
                    ),
                )

                self.assertAllClose(
                    getattr(combined_recording, frame_name).stream.x,
                    np.concatenate(
                        [
                            getattr(recording_1, frame_name).x,
                            getattr(recording_2, frame_name).x,
                        ]
                    ),
                )

                self.assertAllClose(
                    getattr(combined_recording, frame_name).stream.y,
                    np.concatenate(
                        [
                            getattr(recording_1, frame_name).y,
                            getattr(recording_2, frame_name).y,
                        ]
                    ),
                )

                self.assertAllClose(
                    getattr(combined_recording, frame_name).stream.z,
                    np.concatenate(
                        [
                            getattr(recording_1, frame_name).z,
                            getattr(recording_2, frame_name).z,
                        ]
                    ),
                )

            # gps
            self.assertAllClose(
                combined_recording.gps.stream.speed,
                np.concatenate([recording_1.gps.speed, recording_2.gps.speed]),
            )

            self.assertAllClose(
                combined_recording.gps.stream.bearing,
                np.concatenate([recording_1.gps.bearing, recording_2.gps.bearing]),
            )

            self.assertAllClose(
                combined_recording.gps.stream.latitude,
                np.concatenate([recording_1.gps.latitude, recording_2.gps.latitude]),
            )

            # frame streams
            for frame_name in ["dist", "tail"]:
                self.assertAllClose(
                    getattr(combined_recording, frame_name).stream.score,
                    np.concatenate(
                        [
                            getattr(recording_1, frame_name).score,
                            getattr(recording_2, frame_name).score,
                        ]
                    ),
                )

                self.assertAllEqual(
                    getattr(combined_recording, frame_name).stream.system_ms,
                    np.concatenate(
                        [
                            getattr(recording_1, frame_name).system_ms,
                            getattr(recording_2, frame_name).system_ms,
                        ]
                    ),
                )

                self.assertAllEqual(
                    getattr(combined_recording, frame_name).stream.sensor_ns,
                    np.concatenate(
                        [
                            getattr(recording_1, frame_name).sensor_ns,
                            getattr(recording_2, frame_name).sensor_ns,
                        ]
                    ),
                )

            # AQBitStream
            self.assertAllClose(
                combined_recording.aqbit.stream.heading,
                np.concatenate([recording_1.aqbit.heading, recording_2.aqbit.heading]),
            )

            self.assertAllClose(
                combined_recording.aqbit.stream.dt,
                np.concatenate([recording_1.aqbit.dt, recording_2.aqbit.dt]),
            )

            # EKF
            self.assertAllClose(
                combined_recording.ekf.stream.acc_x,
                np.concatenate([recording_1.ekf.acc_x, recording_2.ekf.acc_x]),
            )

            self.assertAllClose(
                combined_recording.ekf.stream.mag_x,
                np.concatenate([recording_1.ekf.mag_x, recording_2.ekf.mag_x]),
            )

            # EKF Config
            self.assertEqual(len(combined_recording.ekf_configs), 2)
            for i in range(2):
                first_dict = combined_recording.ekf_configs[i]._asdict()
                second_dict = [recording_1, recording_2][i].ekf_config._asdict()
                map(first_dict.pop, ["config_x", "config_p", "config_r"])
                map(second_dict.pop, ["config_x", "config_p", "config_r"])
                self.assertDictEqual(first_dict, second_dict)

                self.assertAllClose(
                    combined_recording.ekf_configs[i].config_x,
                    [recording_1, recording_2][i].ekf_config.config_x,
                )

            # Obd
            self.assertAllClose(
                combined_recording.obd.stream.value,
                np.concatenate([recording_1.obd.value, recording_2.obd.value]),
            )

            self.assertAllEqual(
                combined_recording.obd.stream.brick_code,
                np.concatenate(
                    [recording_1.obd.brick_code, recording_2.obd.brick_code]
                ),
            )

            # device_orientation
            self.assertAllEqual(
                combined_recording.device_orientation.stream.name,
                np.concatenate(
                    [
                        recording_1.device_orientation.name,
                        recording_2.device_orientation.name,
                    ]
                ),
            )

            # loose device
            self.assertAllClose(
                combined_recording.loose_device.stream.heuristic_score,
                np.concatenate(
                    [
                        recording_1.loose_device.heuristic_score,
                        recording_2.loose_device.heuristic_score,
                    ]
                ),
            )

            # distraction
            self.assertAllClose(
                combined_recording.dist_multilabel.stream.score_holding_object,
                np.concatenate(
                    [
                        recording_1.dist_multilabel.score_holding_object,
                        recording_2.dist_multilabel.score_holding_object,
                    ]
                ),
            )

            # bounding boxes
            self.assertEqual(
                combined_recording.bounding_boxes_external.stream._size(),
                bb_s_size_1 + bb_s_size_2,
            )
            self.assertAllEqual(
                combined_recording.bounding_boxes_external.stream.sensor_ns,
                np.concatenate(
                    [
                        recording_1.bounding_boxes_external.sensor_ns,
                        recording_2.bounding_boxes_external.sensor_ns,
                    ]
                ),
            )
            for i in range(bb_s_size_1):
                self.assertEqual(
                    combined_recording.bounding_boxes_external.stream.bounding_box[
                        i
                    ]._size(),
                    bb_s_counts_1[i],
                )
                self.assertAllClose(
                    combined_recording.bounding_boxes_external.stream.bounding_box[
                        i
                    ].score,
                    recording_1.bounding_boxes_external.bounding_box[i].score,
                )
                self.assertAllClose(
                    combined_recording.bounding_boxes_external.stream.bounding_box[
                        i
                    ].top,
                    recording_1.bounding_boxes_external.bounding_box[i].top,
                )
                self.assertAllEqual(
                    combined_recording.bounding_boxes_external.stream.bounding_box[
                        i
                    ].objectType,
                    recording_1.bounding_boxes_external.bounding_box[i].objectType,
                )
            for i in range(bb_s_size_2):
                self.assertEqual(
                    combined_recording.bounding_boxes_external.stream.bounding_box[
                        bb_s_size_1 + i
                    ]._size(),
                    bb_s_counts_2[i],
                )
                self.assertAllClose(
                    combined_recording.bounding_boxes_external.stream.bounding_box[
                        bb_s_size_1 + i
                    ].score,
                    recording_2.bounding_boxes_external.bounding_box[i].score,
                )
                self.assertAllClose(
                    combined_recording.bounding_boxes_external.stream.bounding_box[
                        bb_s_size_1 + i
                    ].top,
                    recording_2.bounding_boxes_external.bounding_box[i].top,
                )
                self.assertAllEqual(
                    combined_recording.bounding_boxes_external.stream.bounding_box[
                        bb_s_size_1 + i
                    ].objectType,
                    recording_2.bounding_boxes_external.bounding_box[i].objectType,
                )

            # tailgating
            self.assertAllEqual(
                combined_recording.tailgating.stream.front_box_index,
                np.concatenate(
                    [
                        recording_1.tailgating.front_box_index,
                        recording_2.tailgating.front_box_index,
                    ]
                ),
            )

            # imu_statistics
            self.assertAllEqual(
                combined_recording.imu_statistics.stream.accel_sensor_ns,
                np.concatenate(
                    [
                        recording_1.imu_statistics.accel_sensor_ns,
                        recording_2.imu_statistics.accel_sensor_ns,
                    ]
                ),
            )
            self.assertAllClose(
                combined_recording.imu_statistics.stream.gx_first_moment,
                np.concatenate(
                    [
                        recording_1.imu_statistics.gx_first_moment,
                        recording_2.imu_statistics.gx_first_moment,
                    ]
                ),
            )

            for field_name, field_t in combined_recording._field_types.items():
                if inspect.isclass(field_t) and issubclass(field_t, streams.CombinedStreamMixin):
                    field_val = getattr(combined_recording, field_name)
                    self.assertAllEqual(
                        field_val.lengths,
                        [
                            getattr(recording_1, field_name)._size(),
                            getattr(recording_2, field_name)._size(),
                        ],
                    )

            # model_definitions
            defs_1 = set(recording_1.model_definition)
            defs_2 = set(recording_2.model_definition)
            unique_defs = defs_1.union(defs_2)
            self.assertSetEqual(unique_defs, set(combined_recording.model_definition))

    def test_utc_time_conversion(self):
        file_path_1 = (
            SENSOR_DATA_DIR
            / "2_11_2_imu_statistics"
            / "2a4dcb0a0519e9a7-2019-03-27-sensor-2774c23b3d016d3209a58cf83586702440c39d7e.gz"
        )
        file_path_2 = (
            SENSOR_DATA_DIR
            / "2_11_2_imu_statistics"
            / "2a4dcb0a0519e9a7-2019-03-27-sensor-4b06fee6d55a3a781f26f420dadf70f4fc16cc70.gz"
        )

        r_pb_1 = protobuf.parse_message_from_gzipped_file(
            sensor_pb2.Recording, file_path_1
        )
        r_pb_2 = protobuf.parse_message_from_gzipped_file(
            sensor_pb2.Recording, file_path_2
        )

        recording_1 = sensors.Recording.from_pb(r_pb_1)
        recording_2 = sensors.Recording.from_pb(r_pb_2)
        if (
            recording_1.metadata.utc_boot_time_ns
            == recording_2.metadata.utc_boot_time_ns
        ):
            recording_2 = recording_2._replace(
                metadata=recording_2.metadata._replace(
                    utc_boot_time_ns=recording_2.metadata.utc_boot_time_ns
                    + np.uint64(1)
                )
            )
        if (
            recording_1.metadata.utc_boot_time_offset_ns
            == recording_2.metadata.utc_boot_time_offset_ns
        ):
            recording_2 = recording_2._replace(
                metadata=recording_2.metadata._replace(
                    utc_boot_time_offset_ns=recording_2.metadata.utc_boot_time_offset_ns
                    + np.int64(1)
                )
            )

        combined_recording = sensors.CombinedRecording.from_recordings(
            [recording_1, recording_2]
        )

        utc_combined_recording = combined_recording.to_utc_time()
        for name, field_t in sensors.CombinedRecording._field_types.items():
            if inspect.isclass(field_t) and issubclass(field_t, sensors.CombinedUtcTimeConvertible):
                ns_names = []
                if issubclass(
                    getattr(combined_recording, name).stream.__class__,
                    sensors.SensorStream,
                ):
                    ns_names = ["sensor_ns"]
                elif isinstance(
                    getattr(combined_recording, name).stream,
                    sensors.ImuStatisticsStream,
                ):
                    ns_names = ["accel_sensor_ns", "gyro_sensor_ns"]

                orig_com_stream = getattr(combined_recording, name)
                new_com_stream = getattr(utc_combined_recording, name)
                sizes = orig_com_stream.lengths
                for ns_name in ns_names:
                    self.assertAllEqual(
                        getattr(new_com_stream.stream, ns_name)[: sizes[0]],
                        nauto_trans.to_utc_time(
                            getattr(orig_com_stream.stream, ns_name)[: sizes[0]],
                            recording_1.metadata.utc_boot_time_ns,
                            recording_1.metadata.utc_boot_time_offset_ns,
                        ),
                    )
                    self.assertAllEqual(
                        getattr(new_com_stream.stream, ns_name)[sizes[0] :],
                        nauto_trans.to_utc_time(
                            getattr(orig_com_stream.stream, ns_name)[sizes[0] :],
                            recording_2.metadata.utc_boot_time_ns,
                            recording_2.metadata.utc_boot_time_offset_ns,
                        ),
                    )

    def test_model_definition_merging(self):
        file_paths = [
            SENSOR_DATA_DIR
            / "2_12_distraction_multilabel"
            / "03799-imu-20190429-134003-0700.pb.gz",
            SENSOR_DATA_DIR
            / "2_12_distraction_multilabel"
            / "03805-imu-20190429-134103-0700.pb.gz",
            SENSOR_DATA_DIR
            / "2_12_distraction_multilabel"
            / "03808-imu-20190429-134136-0700.pb.gz",
            SENSOR_DATA_DIR
            / "2_12_distraction_multilabel"
            / "03810-imu-20190429-134154-0700.pb.gz",
        ]
        recordings = [
            sensors.Recording.from_pb(
                protobuf.parse_message_from_gzipped_file(
                    sensor_pb2.Recording, file_path
                )
            )
            for file_path in file_paths
        ]

        # change models
        models = [
            [
                sensors.ModelDefinition("model_1", "1"),
                sensors.ModelDefinition("model_1", "2"),
                sensors.ModelDefinition("model_3", "2"),
                sensors.ModelDefinition("model_2", "1"),
            ],
            [
                sensors.ModelDefinition("model_3", "1"),
                sensors.ModelDefinition("model_1", "2"),
                sensors.ModelDefinition("model_2", "1"),
            ],
            [
                sensors.ModelDefinition("model_1", "1"),
                sensors.ModelDefinition("model_2", "2"),
                sensors.ModelDefinition("model_3", "2"),
                sensors.ModelDefinition("model_5", "0"),
            ],
            [
                sensors.ModelDefinition("model_1", "1"),
                sensors.ModelDefinition("model_3", "1"),
                sensors.ModelDefinition("model_3", "2"),
                sensors.ModelDefinition("model_4", "1"),
            ],
        ]
        unique_models = set([m_def for models_list in models for m_def in models_list])
        recordings = [
            rec._replace(
                model_definition=model_definition,
                bounding_boxes_external=rec.bounding_boxes_external._replace(
                    model_id=np.random.randint(
                        0,
                        len(model_definition),
                        size=rec.bounding_boxes_external._size(),
                        dtype=np.int32,
                    )
                ),
            )
            for rec, model_definition in zip(recordings, models)
        ]
        combined_recording = sensors.CombinedRecording.from_recordings(recordings)
        self.assertSetEqual(unique_models, set(combined_recording.model_definition))

        for bboxes_substr, rec in zip(
            combined_recording.bounding_boxes_external._substreams(), recordings
        ):
            self.assertListEqual(
                [
                    combined_recording.model_definition[m_id]
                    for m_id in bboxes_substr.model_id
                ],
                [
                    rec.model_definition[m_id]
                    for m_id in rec.bounding_boxes_external.model_id
                ],
            )

    def test_from_whole_msg(self):
        for dir_name in [
            "2_10_1_quaternion_w",
            "2_11_2_imu_statistics",
            "2_10_1_oriented_data",
            "2_11_bboxes_tailgating",
            "2_12_tailgating",
            "2_12_distraction_multilabel",
        ]:
            files = list((SENSOR_DATA_DIR / dir_name).iterdir())
            files = sorted(files, key=lambda p: p.name)

            proto_msgs = [
                protobuf.parse_message_from_gzipped_file(
                    sensor_pb2.Recording, file_path
                )
                for file_path in files
            ]
            recordings = [sensors.Recording.from_pb(r_pb) for r_pb in proto_msgs]

            sensors.CombinedRecording.from_recordings(recordings)


class XYZStreamxTest(tf.test.TestCase):
    def get_test_stream(self) -> sensors.XYZStream:
        return sensors.XYZStream(
            x=np.arange(-10, 11),
            y=np.arange(-10, 11),
            z=np.arange(-10, 11),
            w=np.arange(-10, 11),
            sensor_ns=np.arange(-10, 11),
            system_ms=np.arange(-10, 11)
        )

    def test_clip_one(self):
        stream = self.get_test_stream().clip_x(5)
        assert np.max(stream.x) == 5
        assert np.min(stream.x) == -5
        assert np.max(stream.y) == 10
        assert np.min(stream.y) == -10
        assert np.max(stream.z) == 10
        assert np.min(stream.z) == -10

    def test_clip_all(self):
        stream = self.get_test_stream().clip(5)
        assert np.max(stream.x) == 5
        assert np.min(stream.x) == -5
        assert np.max(stream.y) == 5
        assert np.min(stream.y) == -5
        assert np.max(stream.z) == 5
        assert np.min(stream.z) == -5

    def test_cut_end(self):
        stream = self.get_test_stream().cut(8)
        assert stream.x.shape[0] == 8
        assert stream.y.shape[0] == 8
        assert stream.z.shape[0] == 8
        assert stream.w.shape[0] == 8
        assert stream.sensor_ns.shape[0] == 8
        assert stream.system_ms.shape[0] == 8
        assert np.min(stream.x) == -10
        assert np.max(stream.x) == -3

    def test_cut_beginning(self):
        stream = self.get_test_stream().cut(8, cut_beginning=True)
        assert stream.x.shape[0] == 8
        assert stream.y.shape[0] == 8
        assert stream.z.shape[0] == 8
        assert stream.w.shape[0] == 8
        assert stream.sensor_ns.shape[0] == 8
        assert stream.system_ms.shape[0] == 8
        assert np.min(stream.x) == 3
        assert np.max(stream.x) == 10

    def test_cut_too_much(self):
        stream = self.get_test_stream().cut(30)
        assert stream.x.shape[0] == 21
        assert stream.y.shape[0] == 21
        assert stream.z.shape[0] == 21
        assert stream.w.shape[0] == 21
        assert stream.sensor_ns.shape[0] == 21
        assert stream.system_ms.shape[0] == 21
        assert np.min(stream.x) == -10
        assert np.max(stream.x) == 10
