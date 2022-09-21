import abc
import json
from typing import Union, Optional, List

import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import DataFrame, SparkSession

from nauto_datasets import qubole_utils as qu

SENSORS = 'sensors'
SENSOR_FILES = 'sensor_files'
VIDEOS = 'videos'
VIDEO_OUT = 'video_out'
VIDEO_IN = 'video_in'


class DataProvider:

    def __init__(self, spark: SparkSession):
        self._spark = spark

    def get_vehicle_type_mapping(self, query: str) -> pyspark.sql.dataframe.DataFrame:
        return self._spark.sql(query)

    def get_events(self, query: str) -> pyspark.sql.dataframe.DataFrame:
        return self._spark.sql(query)

    def get_videos(self, query: str) -> pyspark.sql.dataframe.DataFrame:
        return self._spark.sql(query)

    def get_sensors(self, query: str) -> pyspark.sql.dataframe.DataFrame:
        return self._spark.sql(query)


class Data(abc.ABC):
    def __init__(self):
        self._content = None

    def get(self) -> pd.DataFrame:
        if self._content is None:
            self._content = self._retrieve()
        return self._content

    @abc.abstractmethod
    def _retrieve(self) -> Union[pd.DataFrame, DataFrame]:
        pass


class VehicleProfileTypeMapping(Data):

    def __init__(self, data_provider: DataProvider):
        super().__init__()
        self._data_provider = data_provider

    @staticmethod
    def get_vehicle_profile_type(row, type_mapping):
        """Match a make/model to vehicle profile."""
        if pd.isnull(row.make) | pd.isnull(row.model):
            return None
        matched_type = type_mapping.loc[(type_mapping.vp_make == row.make.lower().replace(' ', '_')) &
                                        (type_mapping.vp_model == row.model.lower().replace(' ', '_'))]
        if len(matched_type) > 0:
            return matched_type.iloc[0].vehicle_type
        else:
            return None

    def _retrieve(self) -> pd.DataFrame:
        query = "select * from dimension.vehicle_type_mapping"
        vehicle_profile_type_mapping = self._data_provider.get_vehicle_type_mapping(query).toPandas()
        vehicle_profile_type_mapping.columns = ['vp_make', 'vp_model', 'vehicle_type']
        vehicle_profile_type_mapping.vehicle_type = vehicle_profile_type_mapping.vehicle_type.apply(
            lambda x: str(x).replace(' ', '_'))
        ignore_types = ['bike', 'sports_atv', 'utv', 'trailer']
        vehicle_profile_type_mapping = (vehicle_profile_type_mapping
                                        .loc[
                                            ~vehicle_profile_type_mapping.vehicle_type.isin(
                                                ['bad_data', 'nan'] + ignore_types)]
                                        .reset_index(drop=True))
        vehicle_profile_type_mapping = vehicle_profile_type_mapping.applymap(str).applymap(str.lower)

        return vehicle_profile_type_mapping


class PandasDataProvider(DataProvider):

    def __init__(self, spark: SparkSession, frames: List[pd.DataFrame]):
        self._frames = frames
        self._spark = spark

    def get_vehicle_type_mapping(self, query: str) -> pyspark.sql.dataframe.DataFrame:
        return self._spark.createDataFrame(self._frames[0])

    def get_events(self, query: str) -> pyspark.sql.dataframe.DataFrame:
        return self._spark.createDataFrame(self._frames[1])

    def get_videos(self, query: str) -> pyspark.sql.dataframe.DataFrame:
        return self._spark.createDataFrame(self._frames[2])

    def get_sensors(self, query: str) -> pyspark.sql.dataframe.DataFrame:
        return self._spark.createDataFrame(self._frames[3])


class Events(Data, abc.ABC):

    def __init__(self, query: Optional[str] = None, data_provider: Optional[DataProvider] = None,
                 spark: Optional[SparkSession] = None, data_frame: Optional[pd.DataFrame] = None):
        super().__init__()
        self._query = query

        if data_provider is not None:
            self._data_provider = data_provider
        elif spark is not None:
            if data_frame is not None:
                self._data_provider = PandasDataProvider(spark, [None, data_frame, None, None])
            else:
                self._data_provider = DataProvider(spark)
        else:
            raise ValueError('Must provide one: data provider or spark session or data frame')

    @staticmethod
    def _get_sensor_time(params):
        p = json.loads(params)
        if 'abcc_data' in p.keys():
            if ('utc_boot_time_ns' in p.keys()) & ('utc_boot_time_offset_ns' in p.keys()):
                utc_basetime = int(p['utc_boot_time_ns']) + int(p['utc_boot_time_offset_ns'])
            else:
                return (None, None, None)
            if ('event_start_sensor_ns' in p['abcc_data'].keys()) & ('event_end_sensor_ns' in p['abcc_data'].keys()):
                return (utc_basetime,
                        int((utc_basetime + int(p['abcc_data']['event_start_sensor_ns'])) / 1e6),
                        int((utc_basetime + int(p['abcc_data']['event_end_sensor_ns'])) / 1e6))
            else:
                return (None, None, None)
        elif 'crashnet_data' in p.keys():
            if ('utc_boot_time_ns' in p.keys()) & ('utc_boot_time_offset_ns' in p.keys()):
                utc_basetime = int(p['utc_boot_time_ns']) + int(p['utc_boot_time_offset_ns'])
            else:
                return (None, None, None)
            if ('severe_g_event_start_sensor_ns' in p['crashnet_data'].keys()) & (
                    'severe_g_event_end_sensor_ns' in p['crashnet_data'].keys()):
                return (utc_basetime,
                        int((utc_basetime + int(p['crashnet_data']['severe_g_event_start_sensor_ns'])) / 1e6),
                        int((utc_basetime + int(p['crashnet_data']['severe_g_event_end_sensor_ns'])) / 1e6))
            else:
                return (None, None, None)
        else:
            # Old Severe-G algorithm (did not include boot time or start/end. We'll just use +/- 1 second)
            if ('event_start_sensor_time' in p.keys()) & ('event_end_sensor_time' in p.keys()):
                if ('utc_boot_time_ns' in p.keys()) & ('utc_boot_time_offset_ns' in p.keys()):
                    utc_basetime = int(p['utc_boot_time_ns']) + int(p['utc_boot_time_offset_ns'])
                else:
                    utc_basetime = None
                return (utc_basetime, p['peak_time'] - 1000, p['peak_time'] + 1000)
            else:
                return (None, None, None)

    @abc.abstractmethod
    def _retrieve(self) -> pd.DataFrame:
        pass

    @property
    def query(self) -> Optional[str]:
        return self._query


class PackagedEvents(Events):
    def _retrieve(self) -> pd.DataFrame:
        events = self._data_provider.get_events(self._query).toPandas()

        vehicle_profile_type_mapping = VehicleProfileTypeMapping(self._data_provider)
        if 'make' in events and 'model' in events:
            events['vehicle_type'] = events.apply(vehicle_profile_type_mapping.get_vehicle_profile_type,
                                                  type_mapping=vehicle_profile_type_mapping.get(),
                                                  axis=1)
        events[VIDEO_IN] = events['params'].apply(lambda x: [] if x is None else json.loads(x).get(VIDEO_IN, []))
        events[VIDEO_OUT] = events['params'].apply(lambda x: [] if x is None else json.loads(x).get(VIDEO_OUT, []))
        events[VIDEOS] = events.apply(qu.get_list_columns_unique_values, cols=[VIDEO_IN, VIDEO_OUT], axis=1)
        events.videos = events.videos.apply(lambda x: [str(i) for i in x])

        events[SENSOR_FILES] = events['params'].apply(
            lambda x: [] if x is None else json.loads(x).get(SENSOR_FILES, []))
        events[SENSORS] = events.apply(qu.get_list_columns_unique_values, cols=[SENSOR_FILES], axis=1)
        events.sensors = events.sensors.apply(lambda x: [str(i) for i in x])

        results = events.params.apply(self._get_sensor_time)
        events['utc_basetime'] = results.apply(lambda x: x[0])
        events['event_start_ms'] = results.apply(lambda x: x[1])
        events['event_end_ms'] = results.apply(lambda x: x[2])
        events['event_duration_ms'] = events.event_end_ms - events.event_start_ms
        events['event_start'] = pd.to_datetime(events.event_start_ms, unit='ms')
        events['event_end'] = pd.to_datetime(events.event_end_ms, unit='ms')
        events['event_id'] = events.message_id
        events['event_message_type'] = events.apply(lambda x: '{}'.format(x.type), axis=1)
        events['event_message_params'] = (events
                                          .apply(lambda x: '{{"{}": {}}}'.format(x.type, x.params),
                                                 axis=1))

        # Get string version of event media for joining in Spark (not the best way to do it, but it works for now)
        events['videos_str'] = events.videos.apply(lambda x: ','.join(x))
        events['sensors_str'] = events.sensors.apply(lambda x: ','.join(x))

        # Cut off videos after 1 min or 12 segments (likely extra long due to loose device issues)
        events.videos = events.videos.apply(lambda x: x[0:12])
        events.sensors = events.sensors.apply(lambda x: x[0:24])

        # Get timestamps to allow us to capture full event context
        events['request_start_ms'] = events.event_start_ms
        events['request_end_ms'] = events.event_end_ms
        events['request_duration_ms'] = events.request_end_ms - events.request_start_ms
        events['message_params'] = events['params']
        events = events.drop(columns='params')

        return events


class DRTEvents(Events):
    def _retrieve(self) -> pd.DataFrame:
        events = self._data_provider.get_events(self._query).toPandas()

        vehicle_profile_type_mapping = VehicleProfileTypeMapping(self._data_provider)
        if 'make' in events and 'model' in events:
            events['vehicle_type'] = events.apply(vehicle_profile_type_mapping.get_vehicle_profile_type,
                                                  type_mapping=vehicle_profile_type_mapping.get(),
                                                  axis=1)
        if VIDEO_IN not in events:
            events[VIDEO_IN] = np.empty((len(events), 0)).tolist()
        if VIDEO_OUT not in events:
            events[VIDEO_OUT] = np.empty((len(events), 0)).tolist()
        if SENSOR_FILES not in events:
            events[SENSOR_FILES] = np.empty((len(events), 0)).tolist()

        events[VIDEOS] = events.apply(qu.get_list_columns_unique_values, cols=[VIDEO_IN, VIDEO_OUT], axis=1)
        events.videos = events.videos.apply(lambda x: [str(i) for i in x])
        events[SENSORS] = events.apply(qu.get_list_columns_unique_values, cols=[SENSOR_FILES], axis=1)
        events.sensors = events.sensors.apply(lambda x: [str(i) for i in x])

        results = events.message_params.apply(self._get_sensor_time)
        events['utc_basetime'] = results.apply(lambda x: x[0])
        events['event_start_ms'] = results.apply(lambda x: x[1])
        events['event_end_ms'] = results.apply(lambda x: x[2])
        events['event_duration_ms'] = events.event_end_ms - events.event_start_ms
        events['event_start'] = pd.to_datetime(events.event_start_ms, unit='ms')
        events['event_end'] = pd.to_datetime(events.event_end_ms, unit='ms')
        events['event_id'] = events.message_id
        events['event_message_type'] = events.apply(lambda x: '{}'.format(x.message_type), axis=1)
        events['event_message_params'] = (events
                                          .apply(lambda x: '{{"{}": {}}}'.format(x.message_type, x.message_params),
                                                 axis=1))

        # Get string version of event media for joining in Spark (not the best way to do it, but it works for now)
        events['videos_str'] = events.videos.apply(lambda x: ','.join(x))
        events['sensors_str'] = events.sensors.apply(lambda x: ','.join(x))

        # Cut off videos after 1 min or 12 segments (likely extra long due to loose device issues)
        events.videos = events.videos.apply(lambda x: x[0:12])
        events.sensors = events.sensors.apply(lambda x: x[0:24])

        # Get timestamps to allow us to capture full event context
        events['request_start_ms'] = events.event_start_ms
        events['request_end_ms'] = events.event_end_ms
        events['request_duration_ms'] = events.request_end_ms - events.request_start_ms

        return events


class Videos(Data):
    def __init__(self, min_date: str, max_date: str, device_ids: np.ndarray,
                 data_provider: DataProvider):
        super().__init__()
        self._min_date = min_date
        self._max_date = max_date
        self._device_ids = device_ids
        self._data_provider = data_provider

    def _retrieve(self) -> pd.DataFrame:
        videos_query = """
              SELECT *,
                     'video' AS media_type,
                     get_json_object(message_params, '$.version') AS version,
                     cast(get_json_object(message_params, '$.is_sensor_time') AS boolean) AS is_sensor_time,
                     cast(get_json_object(message_params, '$.sensor_start') AS long) AS sensor_start,
                     cast(get_json_object(message_params, '$.sensor_end') AS long) AS sensor_end,
                     (cast(get_json_object(message_params, '$.sensor_end') AS long) - 
                      cast(get_json_object(message_params, '$.sensor_start') AS long)) / 1e9 AS sensor_duration
                FROM device.video_ts
              WHERE message_day BETWEEN '{min_date}' AND '{max_date}'
                 AND device_id IN {device_ids}
            ORDER BY device_id, message_id ASC
            """.format(min_date=self._min_date, max_date=self._max_date,
                       device_ids=qu.create_sql_str(self._device_ids))
        videos = self._data_provider.get_videos(videos_query)

        return videos


class Sensors(Data):

    def __init__(self, min_date: str, max_date: str, device_ids, sensor_msg_ids, data_provider: DataProvider):
        super().__init__()
        self._min_date = min_date
        self._max_date = max_date
        self._device_ids = device_ids
        self._sensor_msg_ids = sensor_msg_ids
        self._data_provider = data_provider

    def _retrieve(self) -> pd.DataFrame:
        sensors_query = """
                              SELECT *,
                                     'sensor' AS media_type
                                FROM device.sensor
                              WHERE message_day BETWEEN '{min_date}' AND '{max_date}'
                                 AND device_id IN {device_ids}
                                 AND message_id IN {msg_ids}
                            ORDER BY device_id, message_id ASC
                            """.format(min_date=self._min_date, max_date=self._max_date,
                                       device_ids=qu.create_sql_str(self._device_ids),
                                       msg_ids=qu.create_sql_str(self._sensor_msg_ids))
        sensors = self._data_provider.get_sensors(sensors_query).drop('sensor_start_timestamp',
                                                                      'sensor_end_timestamp')
        return sensors


class RequestedEvents(Data):
    def __init__(self, events: Events, spark: SparkSession):
        super().__init__()
        self._spark = spark
        self._events = events

    def _retrieve(self) -> Union[pd.DataFrame, DataFrame]:
        events = self._events.get().dropna(subset=['request_duration_ms'], inplace=False)
        requested_events = self._spark.createDataFrame(events.loc[:,
                                                       ['event_id', 'event_start', 'event_end',
                                                        'device_id',
                                                        'fleet_id', 'sensors_str', 'videos_str']]) \
            .select('event_id', 'event_start', 'event_end', 'device_id', 'fleet_id',
                    'videos_str',
                    'sensors_str') \
            .withColumnRenamed('device_id', 'event_device_id') \
            .withColumnRenamed('fleet_id', 'event_fleet_id')
        return requested_events


class RequestedEventsVideos(Data):
    def __init__(self, events: Events, requested_events: RequestedEvents,
                 data_provider: DataProvider):
        super().__init__()
        self._requested_events = requested_events
        self._events = events
        self._data_provider = data_provider

    def _retrieve(self) -> Union[pd.DataFrame, DataFrame]:
        events = self._events.get().dropna(subset=['request_duration_ms'], inplace=False)
        min_date = (events.event_start.min() - pd.Timedelta(1, 'm')).date().strftime('%Y-%m-%d')
        max_date = (events.event_end.max() + pd.Timedelta(1, 'm')).date().strftime('%Y-%m-%d')
        device_ids = events.device_id.unique()
        videos = Videos(min_date, max_date, device_ids, self._data_provider).get()

        join_cond = [self._requested_events.get().event_device_id == videos.device_id,
                     self._requested_events.get().videos_str.contains(videos.message_id)]

        requested_events_videos = (self._requested_events.get()
                                   .join(videos, join_cond, how='left')
                                   .drop('event_device_id', 'event_fleet_id', 'event_start', 'event_end',
                                         'videos_str',
                                         'sensors_str')
                                   .dropDuplicates(['event_id', 'device_id', 'message_id', 'message_type'])
                                   .toPandas()
                                   .sort_values('message_ts')
                                   .reset_index(drop=True))

        return requested_events_videos


class VideoMessageIds(Data):
    def __init__(self, requested_events_videos: RequestedEventsVideos):
        super().__init__()
        self._requested_events_videos = requested_events_videos

    def _retrieve(self) -> pd.DataFrame:
        video_message_ids = self._requested_events_videos.get() \
            .sort_values('message_id') \
            .groupby('event_id')['message_id'] \
            .apply(list).reset_index().rename(columns={'message_id': VIDEOS})

        return video_message_ids


class RequestedEventsSensors(Data):
    def __init__(self, events: Events, requested_events: RequestedEvents, data_provider: DataProvider):
        super().__init__()
        self._events = events
        self._requested_events = requested_events
        self._data_provider = data_provider

    def _retrieve(self) -> pd.DataFrame:
        requested_events_sensors = self._events.get().dropna(subset=['request_duration_ms'], inplace=False)
        min_date = (requested_events_sensors.event_start.min() - pd.Timedelta(1, 'm')).date().strftime(
            '%Y-%m-%d')
        max_date = (requested_events_sensors.event_end.max() + pd.Timedelta(1, 'm')).date().strftime(
            '%Y-%m-%d')
        device_ids = requested_events_sensors.device_id.unique()
        sensor_msg_ids = list(set(np.concatenate(requested_events_sensors.sensors.values)))

        sensors = Sensors(min_date, max_date, device_ids, sensor_msg_ids, self._data_provider).get()

        join_cond = [self._requested_events.get().event_device_id == sensors.device_id,
                     self._requested_events.get().sensors_str.contains(sensors.message_id)]

        requested_events_sensors = (self._requested_events.get()
                                    .join(sensors, join_cond, how='left')
                                    .drop('event_device_id', 'event_fleet_id', 'event_start',
                                          'event_end',
                                          'videos_str',
                                          'sensors_str')
                                    .dropDuplicates(
            ['event_id', 'device_id', 'message_id', 'message_type'])
                                    .toPandas()
                                    .sort_values('message_ts')
                                    .reset_index(drop=True))

        return requested_events_sensors


class SensorMessageIds(Data):
    def __init__(self, requested_events_sensors: RequestedEventsSensors):
        super().__init__()
        self._requested_events_sensors = requested_events_sensors

    def _retrieve(self) -> Union[pd.DataFrame, DataFrame]:
        sensor_message_ids = self._requested_events_sensors.get() \
            .sort_values('message_id').groupby('event_id')['message_id'] \
            .apply(list).reset_index().rename(columns={'message_id': SENSORS})

        return sensor_message_ids
