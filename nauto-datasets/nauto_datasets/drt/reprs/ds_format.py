import json
import os
from typing import Optional, Dict, Tuple

import boto3
import numpy as np
import pandas as pd
from pyspark.sql.session import SparkSession

import nauto_datasets.drt.reprs.package_events as pe
import nauto_datasets.qubole_utils as qu
import nauto_datasets.timeutils as tu
from nauto_datasets.drt.reprs.ds_config import RequestConfig
from nauto_datasets.drt.reprs.ds_data import Events, RequestedEventsVideos, RequestedEvents, VideoMessageIds, \
    RequestedEventsSensors, SensorMessageIds, DataProvider, SENSORS
from nauto_datasets.drt.reprs.event_set import EventSet, EventSource
from nauto_datasets.drt.reprs.processors import ResultProcessor, SparkResultProcessor


class Source(EventSource):

    def __init__(self, s3_client, project_bucket: str, project_key: str, config: RequestConfig):
        self._s3_client = s3_client
        self._project_bucket = project_bucket
        self._project_key = project_key
        self._config = config

    @property
    def events(self) -> pd.DataFrame:
        requested_events = pe.load_events('requested-events.csv', self._s3_client,
                                          s3_bucket=self._project_bucket,
                                          s3_key=self._project_key,
                                          ts_fields=[])
        return requested_events

    def _requested_events_media(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        requested_events_media = pe.load_events_media_messages('requested-events-videos.csv',
                                                               'requested-events-sensors.csv',
                                                               self._s3_client,
                                                               s3_bucket=self._project_bucket,
                                                               s3_key=self._project_key)
        return requested_events_media

    @property
    def videos(self) -> pd.DataFrame:
        return self._requested_events_media()[0]

    @property
    def sensors(self) -> pd.DataFrame:
        return self._requested_events_media()[1]

    @property
    def request_config(self) -> RequestConfig:
        return self._config

    @property
    def events_metadata(self) -> Dict:
        processed_events_metadata = qu.load_data_from_s3(self._s3_client, self._project_bucket,
                                                         os.path.join(self._project_key,
                                                                      'requested-events-metadata.json'))
        return processed_events_metadata


class DSFormatter:

    def __init__(self, project_name: str, project_bucket: str, project_key: str,
                 spark: SparkSession, s3_resource=None,
                 s3_client=None, in_qubole: bool = False, data_provider: Optional[DataProvider] = None,
                 result_processor: Optional[ResultProcessor] = None):
        self._project_name = project_name
        self._project_bucket = project_bucket
        self._project_key = f'{project_key}/data/interim/{project_name}'
        self._project_output_data_key = f'{project_key}/data/processed/{project_name}'
        self._logs_dir = os.path.join(self._project_output_data_key, 'processed-events-logs')

        self._sc = spark.sparkContext
        self._spark = spark
        self._data_provider = DataProvider(spark) if data_provider is None else data_provider
        self._result_processor = SparkResultProcessor(spark) if result_processor is None else result_processor
        self._s3_resource = boto3.resource("s3") if s3_resource is None else s3_resource
        self._s3_client = boto3.client("s3") if s3_client is None else s3_client
        self._in_qubole = in_qubole

    def get_processed_ts_list(self):
        """Get list of previous requests."""
        suffix = '-processed-events-metadata.json'
        processed_ts_list = []
        try:
            for key in qu.get_matching_s3_keys(bucket=self._project_bucket, prefix=self._logs_dir, suffix=suffix):
                processed_ts_list.append(os.path.basename(key).split(suffix)[0])
            processed_ts_list = sorted(processed_ts_list)
        except:
            pass
        return processed_ts_list

    def load_processed_request(self, processed_ts):
        """Load previously processed request results."""
        for key in qu.get_matching_s3_keys(bucket=self._project_bucket,
                                           prefix=os.path.join(self._logs_dir, processed_ts)):
            if 'processed-events-metadata.json' in key:
                metadata = qu.load_data_from_s3(s3_client=self._s3_client, s3_bucket=self._project_bucket, s3_key=key)
            elif 'processed-events.csv' in key:
                data = qu.load_data_from_s3(s3_client=self._s3_client, s3_bucket=self._project_bucket, s3_key=key,
                                            index_col=0)
                for c in ['message_ts', 'received_ts', 'local_message_timestamp', 'media_proccessing_begin',
                          'media_processing_end']:
                    try:
                        data.loc[:, c] = pd.to_datetime(data.loc[:, c])
                    except:
                        pass
        return metadata, data

    @staticmethod
    def check_existing_media(row):
        """Check if event already has media covering our request range"""

        def check_ms_range(msg_list, start_ms, end_ms, msg_len, tol=0.2):
            if pd.isnull(msg_list).any():
                return False
            msg_len_s = pd.Timedelta(msg_len, 's')
            if len(msg_list) > 0:
                ts_list = sorted([tu.datetime_from_hex_ns(val) for val in msg_list])
                min_ms = tu.ms_from_datetime(ts_list[0])
                max_ms = tu.ms_from_datetime(ts_list[-1] + msg_len_s)
                if len(msg_list) == 1:
                    return (min_ms <= start_ms) & (max_ms >= end_ms)
                else:
                    max_ms_diff = np.max(np.diff(ts_list))
                    return (min_ms <= start_ms) & (max_ms >= end_ms) & (max_ms_diff <= msg_len_s * msg_len * tol)
            else:
                return False

        vid_range_check, sensor_range_check = False, False
        if 'videos' in row.keys():
            vid_range_check = check_ms_range(row['videos'], row['request_start_ms'], row['request_end_ms'], 5)
        if SENSORS in row.keys():
            sensor_range_check = check_ms_range(row[SENSORS], row['request_start_ms'], row['request_end_ms'], 10)

        return (vid_range_check & sensor_range_check)

    def generate_data_set(self, config: RequestConfig, events_object: Events):
        events = events_object.get()
        if config.prepare_new_request:
            if self._in_qubole:
                print("%html <div style='font-size: 18px;'>Processing {} events</div>".format(events.shape[0]))
                qu.show_df(events.event_message_type.value_counts())
                qu.show_df(events, max_height=250)
            else:
                print(f'Processing {events.shape[0]} events')
                print(events.event_message_type.value_counts().iloc[0:100])
                print(events.iloc[0:100])
        else:
            if self._in_qubole:
                print("%html <div style='font-size: 18px;'>PREP_NEW_REQUEST is False. Not requesting new data.</div>")
            else:
                print('prep_new_request is False. Not requesting new data.')

        if config.prepare_new_request:
            preparing_events_start_time = pd.datetime.utcnow()

            requested_events = events.copy()
            requested_events['message_id_response'] = None
            requested_events['media_status'] = None
            requested_events['has_audio'] = None

            cols = ['fleet_id', 'device_id', 'event_id', 'event_message_type', 'message_ts', 'event_duration_ms']
            requested_events = requested_events.loc[:, cols + [c for c in requested_events.columns if c not in cols]]

            if len(requested_events) > 0:
                requested_events = RequestedEvents(events_object, self._spark)

                requested_events_videos = RequestedEventsVideos(events_object, requested_events, self._data_provider)
                video_message_ids = VideoMessageIds(requested_events_videos).get()

                requested_events_sensors = RequestedEventsSensors(events_object, requested_events, self._data_provider)
                sensor_message_ids = SensorMessageIds(requested_events_sensors).get()

                requested_events = events_object.get() \
                    .drop(columns=['videos', 'videos_str', SENSORS, 'sensors_str']) \
                    .dropna(subset=['request_duration_ms'], inplace=False) \
                    .merge(video_message_ids, on='event_id') \
                    .merge(sensor_message_ids, on='event_id')

                # Check if all event media is available
                requested_events['media_available'] = requested_events.apply(DSFormatter.check_existing_media, axis=1)

                # Fix records with no media in list
                def fix_empty_list(x):
                    if type(x) == list:
                        if len(x) == 1:
                            if x[0] is None:
                                return []
                        else:
                            return x
                    elif x is None:
                        return []
                    else:
                        return x

                requested_events.sensors = requested_events.sensors.apply(fix_empty_list)
                requested_events.videos = requested_events.videos.apply(fix_empty_list)

                # Save query results to S3
                #   (Separates SQL queries from processing for easier debugging)
                preparing_events_end_time = pd.datetime.utcnow()
                requested_events_metadata = {'description': config.request_description,
                                             'requested_by': config.requested_by,
                                             'prepared_by': config.prepared_by,
                                             'rights': config.rights,
                                             'dataset_name': config.dataset_name,
                                             'query_info': {'start_ts': str(preparing_events_start_time),
                                                            'end_ts': str(preparing_events_end_time),
                                                            },
                                             'query_results': {'requested_events': requested_events.shape[0],
                                                               'requested_events_videos':
                                                                   requested_events_videos.get().shape[
                                                                       0],
                                                               'requested_events_sensors':
                                                                   requested_events_sensors.get().shape[0],
                                                               'min_event_ts': str(requested_events.message_ts.min()),
                                                               'max_event_ts': str(requested_events.message_ts.max())
                                                               },
                                             }

                _ = qu.save_data_to_s3(requested_events, self._s3_resource, self._project_bucket,
                                       os.path.join(self._project_key, 'requested-events.csv'))
                _ = qu.save_data_to_s3(requested_events_videos.get(), self._s3_resource, self._project_bucket,
                                       os.path.join(self._project_key, 'requested-events-videos.csv'))
                _ = qu.save_data_to_s3(requested_events_sensors.get(), self._s3_resource, self._project_bucket,
                                       os.path.join(self._project_key, 'requested-events-sensors.csv'))
                _ = qu.save_data_to_s3(requested_events_metadata, self._s3_resource, self._project_bucket,
                                       os.path.join(self._project_key, 'requested-events-metadata.json'))

                events_media = pd.concat((requested_events_sensors.get(), requested_events_videos.get()),
                                         sort=False).reset_index(drop=True)
                events_media_to_show = events_media.drop('message_params', axis=1).groupby(
                    ['device_id', 'event_id', 'message_type']).message_id.count().unstack('message_type')
                requested_events_to_show = requested_events.drop(['message_params', 'videos', SENSORS], axis=1)

                if self._in_qubole:
                    qu.show_df(events_media_to_show,
                               limit_results=False, max_height=200, caption='event media found')
                    qu.show_df(requested_events_to_show,
                               limit_results=False, max_height=200, caption='Requested Events')
                    print('%html <div>Requested Events Metadata:<br>{}</div>'.format(
                        json.dumps(requested_events_metadata, indent=4).replace('\n', '<br>').replace(' ', '&nbsp')))
                else:
                    print('event media found')
                    print(events_media_to_show)
                    print('Requested Events')
                    print(requested_events_to_show)
                    print(f'Requested Events Metadata:\n{json.dumps(requested_events_metadata, indent=4)}')
        else:
            if self._in_qubole:
                print("%html <div style='font-size: 18px;'>prep_new_request is False. Not requesting new data.</div>")
            else:
                print('prep_new_request is False. Not requesting new data.')

        if config.prepare_new_request:
            # Processing timestamp for record keeping
            process_ts = pd.datetime.utcnow()

            # Locations to store processed_events data and processing_logs
            output_dir = os.path.join(self._project_output_data_key, 'processed-events',
                                      process_ts.strftime('%Y%m%d_%H%M%S'))

            # Pull list of requested events and associated media from S3
            source = Source(self._s3_client, self._project_bucket, self._project_key, config)
            event_set = EventSet(source, self._project_bucket, self._project_output_data_key, self._project_name, self._spark,
                                 self._s3_resource)
            processed_events, processed_events_metadata = event_set.create(self._result_processor)

            if self._in_qubole:
                if config.dry_run is True:
                    print("%html <div style='font-size: 18px;'>dry_run is True. Dry Run Results:</div>")
                # Show event media issue counts
                qu.show_df(processed_events.check_event_media.value_counts(), caption='Processed events media issues')
                print('%html <div>Processed Events Metadata:<br>{}</div>'.format(
                    json.dumps(processed_events_metadata, indent=4).replace('\n', '<br>').replace(' ', '&nbsp')))
                qu.show_df(processed_events.loc[:, ['event_id', 'event_message_type', 'check_event_media']],
                           max_height=250)
                print("%html <div style='font-size: 14px;'>Processing completed in {}</div>".format(
                    qu.format_datetimes(pd.datetime.utcnow() - process_ts)))
            else:
                if config.dry_run is True:
                    print('dry_run is True. Dry Run Results:')
                print(processed_events.check_event_media.value_counts().iloc[0:100])
                print(f'Processed Events Metadata:\n{json.dumps(processed_events_metadata, indent=4)}')
                print(processed_events.loc[:, ['event_id', 'event_message_type', 'check_event_media']].iloc[0:100])
                print(f'Processing completed in {qu.format_datetimes(pd.datetime.utcnow() - process_ts)}')

            # Store collisions CSV per group and processing record for logging
            if config.dry_run is False:
                event_set.upload()

                request_key = os.path.join(self._logs_dir,
                                           '{}-processed-events-request.json'.format(
                                               process_ts.strftime('%Y%m%d_%H%M%S')))
                request_metadata = {'query': events_object.query, 'config': config.to_dict()}
                _ = qu.save_data_to_s3(request_metadata, self._s3_resource, self._project_bucket, request_key)

                # Display aws cli command for downloading results
                if self._in_qubole:
                    print(
                        "%html <br><div style='font-size: 16px;'>Download dataset:<br>aws s3 sync s3://{}/{} ./{}-{} --profile nauto-prod-us</div>"
                            .format(self._project_bucket, output_dir, process_ts.strftime('%Y%m%d_%H%M%S'),
                                    config.dataset_name))
                else:
                    print(
                        f'Download dataset:\naws s3 sync s3://{self._project_bucket}/{output_dir} ./{process_ts.strftime("%Y%m%d_%H%M%S")}-{config.dataset_name} --profile nauto-prod-us')
        else:
            if self._in_qubole:
                print("%html <div style='font-size: 18px;'>prep_new_request is False. Loading previous results.</div>")
            else:
                print('prep_new_request is False. Loading previous results.')

            processed_ts_list = self.get_processed_ts_list()
            if len(processed_ts_list) == 0:
                if self._in_qubole:
                    print("%html <div style='font-size: 18px;'>No previous results found.</div>")
                else:
                    print('No previous results found.')
            else:
                processed_ts = processed_ts_list[-1]  # most recent request by default
                processed_events_metadata, processed_events = self.load_processed_request(processed_ts)

                if self._in_qubole:
                    print("%html <div style='font-size: 18px;'>Request Processed at {} (UTC)</div>".format(
                        pd.to_datetime(processed_ts, format='%Y%m%d_%H%M%S%f')))
                    qu.show_df(processed_events.check_event_media.value_counts(),
                               caption='Processed events media issues')
                    print('%html <div>Processed Events Metadata:<br>{}</div>'.format(
                        json.dumps(processed_events_metadata, indent=4).replace('\n', '<br>').replace(' ', '&nbsp')))
                    qu.show_df(processed_events.loc[:, ['event_id', 'event_message_type', 'check_event_media']],
                               limit_results=False, max_height=250)
                    # Display aws cli command for downloading results
                    print("%html <div style='font-size: 14px;'>Processing completed in {}</div>"
                        .format(
                        qu.format_datetimes(pd.to_datetime(processed_events_metadata['processing_info']['end_ts']) -
                                            pd.to_datetime(processed_events_metadata['processing_info']['start_ts']))))
                    print(
                        "%html <br><div style='font-size: 16px;'>Download dataset:<br>aws s3 sync s3://{}/{} ./{}_{} --profile nauto-prod-us</div>"
                            .format(processed_events_metadata['processing_info']['s3_bucket'],
                                    processed_events_metadata['processing_info']['output_key'],
                                    processed_events_metadata['processing_info']['output_key'].split('/')[-1],
                                    processed_events_metadata['dataset_name']))
                else:
                    print(f'Request Processed at {pd.to_datetime(processed_ts, format="%Y%m%d_%H%M%S%f")} (UTC)')
                    print(processed_events.check_event_media.value_counts().iloc[0:100])
                    print(f'Processed Events Metadata:\n{json.dumps(processed_events_metadata, indent=4)}')
                    print(processed_events.loc[:, ['event_id', 'event_message_type', 'check_event_media']])
                    # Display aws cli command for downloading results
                    print(f"Processing completed in "
                          f"{qu.format_datetimes(pd.to_datetime(processed_events_metadata['processing_info']['end_ts']) - pd.to_datetime(processed_events_metadata['processing_info']['start_ts']))}")
                    print(f"Download dataset:\naws s3 sync s3://"
                          f"{processed_events_metadata['processing_info']['s3_bucket']}/{processed_events_metadata['processing_info']['output_key']} ./{processed_events_metadata['processing_info']['output_key'].split('/')[-1]}_{processed_events_metadata['dataset_name']} --profile nauto-prod-us")
