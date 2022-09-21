import abc
import logging
from datetime import datetime
from typing import List, NamedTuple, Optional, Dict

from pyspark.sql import DataFrame, SparkSession

from nauto_datasets.core import spark
from nauto_datasets.core.dataset import DataSource
from nauto_datasets.core.schema import RecordSchema
from nauto_datasets.drt import events, judgments, media
from nauto_datasets.drt.judgments import JudgmentSummaryFilter
from nauto_datasets.drt.types import (EventType, JudgmentType, MediaType)
from nauto_datasets.protos import drt_pb2


class DRTConfig(NamedTuple):
    """Determines the rows returned by the `DRTDataSource`.

    Each row consists of:
    * event id (unique key) and event metadata
    * judgments: summaries and infos for each included judgment
    * media: aggregated links for every selected media type
    All information listed above is grouped by the event id

    Attributes:
        judgment_types: types of judgements for which the dataset
            should be created
        event_types: types of events to select
        media_types: the types of media to collect for each event.
        ignore_missing_media: if False then rows for which some
            media files are missing will not be included in the dataset
        judgments_summary_filters: filters judgments based on
            the value of their summary fields.
    """
    judgment_types: List[JudgmentType]
    event_types: Optional[List[EventType]] = None
    media_types: Optional[List[MediaType]] = None
    ignore_missing_media: bool = False
    judgment_summary_filters: Optional[Dict[JudgmentType, JudgmentSummaryFilter]] = None

    def to_pb(self) -> drt_pb2.DRTConfig:
        """Serializes drt config as protobuf `Message`"""
        event_types = self.event_types or []
        media_types = self.media_types or []
        judgment_summary_filters = self.judgment_summary_filters or {}
        return drt_pb2.DRTConfig(
            judgment_types=[jt.to_pb() for jt in self.judgment_types],
            judgment_summary_filters={
                jt.to_pb(): summary_filter.to_pb()
                for jt, summary_filter in judgment_summary_filters.items()
            },
            event_types=[et.to_pb() for et in event_types],
            media_types=[mt.to_pb() for mt in media_types],
            ignore_missing_media=self.ignore_missing_media)

    @staticmethod
    def from_pb(conf_pb: drt_pb2.DRTConfig) -> 'DRTConfig':
        """Deserializes drt config from protobuf `Message`"""
        return DRTConfig(
            judgment_types=[
                JudgmentType.from_pb(jt_pb) for jt_pb in conf_pb.judgment_types
            ],
            judgment_summary_filters={
                JudgmentType.from_pb(jt_pb): JudgmentSummaryFilter.from_pb(summary_filter_pb)
                for jt_pb, summary_filter_pb in conf_pb.judgment_summary_filters.items()
            },
            event_types=[
                EventType.from_pb(et_pb) for et_pb in conf_pb.event_types
            ],
            media_types=[
                MediaType.from_pb(mt_pb) for mt_pb in conf_pb.media_types
            ],
            ignore_missing_media=conf_pb.ignore_missing_media)

    @staticmethod
    def pb_message_type() -> type:
        return drt_pb2.DRTConfig


class DRTDataSource(DataSource):
    """Data Source responsible for producing data based on DRT database.
    Each row should consist of columns belonging to on of three entities:
    - events:
       columns with basic event meta informations, such as event_id, fleet_id,
       or device_id. (look up @{nauto_datasets.drt.events.EventColumns})
    - judgments:
       each judgment type: e.g. COLLISION for SEVERE_G_EVENT event type
       corresponds to two collumns:
          * <judgment_type>_label - a boolean column
          * <judgment_type>_info - a nullable string column with the json
          representing additional info assigned to the judgment
    - media:
       each chosen media type corresponds to two columns
          * <media_type>_message_ids with ordered ids of the messages for each
            media file of this type
          * <media_type>_paths with a list of s3 paths for each media file
    """

    def __init__(self,
                 drt_config: DRTConfig) -> None:
        """Creates a DRT Data Source

        Args:
            drt_config: configuration of the dataset rows
        """
        self._drt_config = drt_config
        # TODO: Move configuration and loggers to other place
        self._logger = logging.getLogger('DRTDataSource')
        self._logger.setLevel(logging.DEBUG)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return (f'DRTDataSource(\n'
                f'drt_config={self._drt_config}')

    def to_pb(self) -> drt_pb2.DRTDataSource:
        """Serializes drt data source as protobuf `Message`"""
        return drt_pb2.DRTDataSource(drt_config=self._drt_config.to_pb())

    @staticmethod
    def from_pb(dds_pb: drt_pb2.DRTDataSource) -> 'DRTConfig':
        """Deserializes drt data source from protobuf `Message`"""
        return DRTDataSource(drt_config=DRTConfig.from_pb(dds_pb.drt_config))

    @staticmethod
    def pb_message_type() -> type:
        return drt_pb2.DRTDataSource

    @property
    def drt_config(self) -> DRTConfig:
        return self._drt_config

    def produce(self,
                sess: SparkSession,
                since: Optional[datetime] = None,
                until: Optional[datetime] = None) -> DataFrame:
        """Returns data for specified time range.

        Args:
            sess: SparkSession
            since: if None, then the data will taken since the beginning
                of time.
            until: if None, then there will be no upperbound on the time
                of each returned row
            kwargs: additional keyword arguments

        Returns:
            a dataframe with dataset examples.
        """
        judgments_sql = judgments.hive_sql_query(
            self._drt_config.judgment_types,
            self._drt_config.judgment_summary_filters,
            self._drt_config.event_types,
            since, until)
        media_sql = media.hive_sql_query(
            self._drt_config.media_types,
            self._drt_config.ignore_missing_media,
            since,
            until)

        self._logger.info(f'JUDGMENTS hive sql query:\n{judgments_sql}')
        self._logger.info(f'MEDIA hive sql query:\n{media_sql}')

        judgments_df = sess.sql(judgments_sql)
        media_df = sess.sql(media_sql)
        return judgments_df.join(
            media_df, events.EventColumns.ID.name, how='inner')

    def record_schema(self) -> RecordSchema:

        events_columns = events.EventColumns.all()

        judgment_types = list(set(self._drt_config.judgment_types))
        judgment_columns = [
            judgments.get_label_column(jt) for jt in judgment_types
        ] + [judgments.get_info_column(jt) for jt in judgment_types]

        media_types = list(set(self._drt_config.media_types))
        media_columns = [
            media.get_ids_column(mt, not self._drt_config.ignore_missing_media)
            for mt in media_types
        ] + [
            media.get_paths_column(
                mt, not self._drt_config.ignore_missing_media)
            for mt in media_types
        ]

        return RecordSchema(
            entities=dict(
                events=events_columns,
                judgments=judgment_columns,
                media=media_columns))
