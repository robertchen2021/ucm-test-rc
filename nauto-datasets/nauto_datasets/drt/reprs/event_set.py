import abc
import os
import tempfile
from typing import Dict, List, Optional
from datetime import datetime

import boto3
import pandas as pd
from pandas import json_normalize
from pyspark.sql import SparkSession

from nauto_datasets.drt.reprs.ds_config import RequestConfig
from nauto_datasets.drt.reprs.ds_data import VIDEOS, SENSORS
import nauto_datasets.qubole_utils as qu
import nauto_datasets.drt.reprs.package_events as pe
from nauto_datasets.drt.reprs.processors import ResultProcessor, SparkResultProcessor


class EventSource(abc.ABC):
    @property
    def videos_column(self) -> str:
        return 'videos'

    @property
    def sensors_column(self) -> str:
        return 'sensors'

    @property
    @abc.abstractmethod
    def events(self) -> pd.DataFrame:
        pass

    @property
    @abc.abstractmethod
    def videos(self) -> pd.DataFrame:
        pass

    @property
    @abc.abstractmethod
    def sensors(self) -> pd.DataFrame:
        pass

    @property
    @abc.abstractmethod
    def request_config(self) -> RequestConfig:
        pass

    @property
    @abc.abstractmethod
    def events_metadata(self) -> Dict:
        pass


class EventSet:

    def __init__(self, source: EventSource, project_bucket: str, project_key: str, project_name: str,
                 spark: SparkSession, s3_resource=None):
        self._spark = spark
        self._project_output_data_key = project_key
        self._project_bucket = project_bucket
        self._source = source
        self._process_ts = datetime.utcnow()
        self._s3_resource = boto3.resource("s3") if s3_resource is None else s3_resource
        self._logs_dir = os.path.join(self._project_output_data_key, 'processed-events-logs')
        self._processed_events = None

    def create(self, result_processor: Optional[ResultProcessor] = None):
        result_processor = result_processor if result_processor is not None else SparkResultProcessor(self._spark)
        events = self._source.events
        videos = self._source.videos
        sensors = self._source.sensors
        videos_column = self._source.videos_column
        sensors_column = self._source.sensors_column
        config = self._source.request_config

        additional_media_columns = []
        if events[videos_column].apply(lambda v: len(v)).max() > 0:
            additional_media_columns.append(videos_column)
        if events[sensors_column].apply(lambda v: len(v)).max() > 0:
            additional_media_columns.append(sensors_column)

        events['media_filename'] = events.apply(lambda x: '{}-{}'.format(x.device_id, x.event_id), axis=1)

        handle_failure_cases = [fc.value for fc in config.handle_failure_cases]

        events_spark = (self._spark.createDataFrame(events.loc[:, ResultProcessor.COLUMNS + additional_media_columns]))

        local_temp_dir = os.path.join(tempfile.gettempdir(), 'events')

        def process_event(event, video_messages=videos,
                          sensor_messages=sensors,
                          s3_bucket=self._project_bucket, s3_output=True, s3_output_dir=self.output_dir,
                          temp_dir=local_temp_dir,
                          join_videos=config.join_videos.value, join_type=config.join_videos_type.value,
                          extract_audio=config.extract_audio,
                          create_sensor_json=config.create_sensor_json, plot_sensor_data=config.plot_sensor_data,
                          metadata_cols=config.metadata_cols, created_by_email=config.prepared_by,
                          rights=config.rights,
                          handle_failure_cases=handle_failure_cases):
            to_dict_op = getattr(event, "asDict", None) or getattr(event, "to_dict", None)
            event_dict = to_dict_op()
            event_dir = '{}-{}'.format(event.device_id, event.event_id)
            s3_output_dir = os.path.join(s3_output_dir, event_dir)
            begin_ts = datetime.utcnow()
            status = pe.process_event_media(event_dict, video_messages, sensor_messages,
                                            s3_bucket=s3_bucket, s3_output=s3_output, s3_output_dir=s3_output_dir,
                                            temp_dir=temp_dir,
                                            join_videos=join_videos, join_type=join_type,
                                            extract_audio=extract_audio,
                                            create_sensor_json=create_sensor_json,
                                            plot_sensor_data=plot_sensor_data,
                                            handle_failure_cases=handle_failure_cases,
                                            media_fname=event_dict['media_filename'],
                                            metadata_cols=metadata_cols, created_by_email=created_by_email,
                                            rights=rights,
                                            dry_run=config.dry_run)
            end_ts = datetime.utcnow()
            status['media_processing_begin'] = str(begin_ts)
            status['media_processing_end'] = str(end_ts)
            status['media_processing_duration'] = str((end_ts - begin_ts).total_seconds())
            status['media_filepath'] = s3_output_dir

            if 'upload_to_s3' in status.keys():
                if 'create_video' in status.keys():
                    if (status['create_video'] == 'success') & (status['upload_to_s3'] == 'success'):
                        status['video_filename'] = '{}.mp4'.format(event_dict['media_filename'])
                if 'create_sensor_json' in status.keys():
                    if (status['create_sensor_json'] == 'success') & (status['upload_to_s3'] == 'success'):
                        status['sensor_json_filename'] = '{}.json.gz'.format(event_dict['media_filename'])
                if 'create_sensor_plot' in status.keys():
                    if (status['create_sensor_plot'] == 'success') & (status['upload_to_s3'] == 'success'):
                        status['sensor_plot_filename'] = '{}.html'.format(event_dict['media_filename'])
            else:
                status['upload_to_s3'] = None

            return status

        process_results = result_processor.process(events_spark, process_event)

        processed_events = events.join(json_normalize(process_results.event_media_check_response))
        process_end_ts = datetime.utcnow()

        processed_events_metadata = self._source.events_metadata

        processed_events_metadata['processing_info'] = {'start_ts': str(self._process_ts),
                                                        'end_ts': str(process_end_ts),
                                                        's3_bucket': self._project_bucket,
                                                        'output_key': self.output_dir,
                                                        'logs_key': self.logs_key,
                                                        'handle_failure_cases': handle_failure_cases,
                                                        'join_videos': config.join_videos.value,
                                                        'join_videos_type': config.join_videos_type.value,
                                                        'extract_audio': str(config.extract_audio),
                                                        'create_sensor_json': str(config.create_sensor_json),
                                                        'create_sensor_plot': str(config.plot_sensor_data),
                                                        }
        processed_events_metadata['processing_results'] = {'events_processed': processed_events.shape[0],
                                                           'event_media_check_status': processed_events.check_event_media.value_counts().to_dict(),
                                                           'upload_status': processed_events.upload_to_s3.value_counts().to_dict()
                                                           }
        self._processed_events = processed_events
        return processed_events, processed_events_metadata

    @property
    def output_dir(self) -> str:
        output_dir = os.path.join(self._project_output_data_key, 'processed-events',
                                  self._process_ts.strftime('%Y%m%d_%H%M%S'))
        return output_dir

    @property
    def logs_key(self) -> str:
        logs_key = os.path.join(self._logs_dir,
                                '{}-processed-events.csv'.format(self._process_ts.strftime('%Y%m%d_%H%M%S')))
        return logs_key

    @property
    def processed_events(self) -> pd.DataFrame:
        return self._processed_events

    @property
    def processed_events_metadata(self) -> Dict:
        return self._source.events_metadata

    def upload(self, processed_events: Optional[pd.DataFrame] = None,
               processed_events_metadata: Optional[pd.DataFrame] = None,
               output_cols: List[str] = ['fleet_id', 'device_id', 'driver_id', 'vehicle_id', 'event_id',
                                         'event_message_type',
                                         'event_start', 'event_end', 'event_duration_ms',
                                         'message_id', 'message_ts', 'message_day', 'message_type', 'message_params'
                                                                                                    'vin', 'make',
                                         'model',
                                         'year',
                                         'vehicle_type',
                                         'utc_basetime', VIDEOS, SENSORS, 'check_event_media',
                                         'handle_check_event_media_failure',
                                         'media_filepath', 'video_filename', 'sensor_json_filename',
                                         'sensor_plot_filename']):
        processed_events = processed_events if processed_events is not None else self.processed_events
        processed_events_metadata = processed_events_metadata if processed_events_metadata is not None else self.processed_events_metadata

        config = self._source.request_config

        metadata_key = os.path.join(self._logs_dir,
                                    '{}-processed-events-metadata.json'.format(
                                        self._process_ts.strftime('%Y%m%d_%H%M%S')))

        processed_events_output = processed_events.loc[processed_events.upload_to_s3 == 'success'].copy() \
            .reset_index(drop=True)
        processed_events_output = processed_events_output.reindex(columns=output_cols)
        output_key = os.path.join(self.output_dir, 'nauto-events-{}.csv'.format(config.dataset_name))
        _ = qu.save_data_to_s3(processed_events_output.reset_index(drop=True), self._s3_resource,
                               self._project_bucket, output_key,
                               index=True)
        _ = qu.save_data_to_s3(processed_events, self._s3_resource, self._project_bucket, self.logs_key, index=True)
        _ = qu.save_data_to_s3(processed_events_metadata, self._s3_resource, self._project_bucket, metadata_key)
