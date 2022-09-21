import unittest

from pyspark.sql import types as dtype

from nauto_datasets.drt import events
from nauto_datasets.drt.data_source import DRTConfig, DRTDataSource
from nauto_datasets.drt.judgments import JudgmentSummaryFilter
from nauto_datasets.drt.types import EventType, JudgmentType, MediaType


class DRTDataSourceTest(unittest.TestCase):
    def test_record_schema(self):
        def test(ignore_missing_media: bool):
            ds = DRTDataSource(
                DRTConfig(
                    judgment_types=[JudgmentType.COLLISION, JudgmentType.DISTRACTION],
                    event_types=[EventType.SEVERE_G_EVENT],
                    media_types=[MediaType.SENSOR, MediaType.VIDEO_IN],
                    ignore_missing_media=ignore_missing_media))

            schema = ds.record_schema()

            self.assertSetEqual(
                set(schema.entities.keys()),
                set(['events', 'judgments', 'media']))

            events_columns = schema.entities['events']
            self.assertSetEqual(
                set(events_columns), set(events.EventColumns.all()))

            judgments_columns = schema.entities['judgments']

            self.assertSetEqual(
                set(judgments_columns),
                set(
                    [
                        dtype.StructField('collision_label',
                                          dtype.BooleanType(), True),
                        dtype.StructField('collision_info', dtype.StringType(),
                                          True),
                        dtype.StructField('distraction_label',
                                          dtype.BooleanType(), True),
                        dtype.StructField('distraction_info',
                                          dtype.StringType(), True)
                    ]))
            media_columns = schema.entities['media']

            required = not ignore_missing_media
            self.assertSetEqual(
                set(media_columns),
                set([
                    dtype.StructField(
                        'sensor_message_ids',
                        dtype.ArrayType(dtype.LongType(), not required),
                        not required),
                    dtype.StructField(
                        'sensor_paths',
                        dtype.ArrayType(dtype.StringType(), not required),
                        not required),
                    dtype.StructField(
                        'video_in_message_ids',
                        dtype.ArrayType(dtype.LongType(), not required),
                        not required),
                    dtype.StructField(
                        'video_in_paths',
                        dtype.ArrayType(dtype.StringType(), not required),
                        not required)
                ]))

        test(True)
        test(False)

    def test_serialization_to_pb(self):
        ds = DRTDataSource(
            DRTConfig(
                judgment_types=[
                    JudgmentType.COLLISION, JudgmentType.OBSTRUCTED_CAMERA
                ],
                media_types=[
                    MediaType.SENSOR, MediaType.VIDEO_IN, MediaType.SNAPSHOT_IN
                ],
                ignore_missing_media=True))

        ds_pb = ds.to_pb()
        des_ds_pb = DRTDataSource.from_pb(ds_pb)

        self.assertEqual(
            des_ds_pb.drt_config,
            ds.drt_config._replace(
                event_types=[], judgment_summary_filters={}))

        ds = DRTDataSource(
            DRTConfig(
                event_types=[EventType.DISTRACTION],
                judgment_types=[JudgmentType.TIMELINE],
                judgment_summary_filters={
                    JudgmentType.TIMELINE: JudgmentSummaryFilter(['true', 'false'], invert=False)
                },
                media_types=[
                    MediaType.SENSOR, MediaType.SNAPSHOT_OUT, MediaType.SNAPSHOT_IN
                ],
                ignore_missing_media=False))

        ds_pb = ds.to_pb()
        des_ds_pb = DRTDataSource.from_pb(ds_pb)

        self.assertEqual(des_ds_pb.drt_config, ds.drt_config)
