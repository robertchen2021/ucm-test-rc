import json
from pathlib import Path

import boto3
import pandas as pd
import pyspark
from moto import mock_s3
from pyspark.sql.session import SparkSession

from nauto_datasets.drt.reprs.ds_config import RequestConfig
from nauto_datasets.drt.reprs.ds_data import DataProvider, DRTEvents, PackagedEvents, VIDEO_IN, VIDEO_OUT, SENSOR_FILES
from nauto_datasets.drt.reprs.ds_format import ResultProcessor, DSFormatter


class StaticDataProvider(DataProvider):

    def _read_from_pickle(self, pickle_type: str) -> pyspark.sql.dataframe.DataFrame:
        path_prefix = str((Path(__file__).parent.parent / 'test_data' / 'ds_data').resolve())
        pickle = pd.read_pickle(f'{path_prefix}/{pickle_type}.pkl')
        return pickle

    def get_vehicle_type_mapping(self, query: str) -> pyspark.sql.dataframe.DataFrame:
        return self._spark.createDataFrame(self._read_from_pickle('vehicle_profile_type_mapping'))

    def get_events(self, query: str) -> pyspark.sql.dataframe.DataFrame:
        pickle = self._read_from_pickle('events')
        pickle['upload_data'] = ''
        pickle['sensor_start'] = ''
        pickle['sensor_end'] = ''
        pickle['bounce'] = ''
        pickle['is_hard'] = ''
        pickle['snapshot_in'] = ''
        pickle['snapshot_out'] = ''
        return self._spark.createDataFrame(pickle)

    def get_videos(self, query: str) -> pyspark.sql.dataframe.DataFrame:
        return self._spark.createDataFrame(self._read_from_pickle('videos'))

    def get_sensors(self, query: str) -> pyspark.sql.dataframe.DataFrame:
        return self._spark.createDataFrame(self._read_from_pickle('sensors'))


class TestResultProcessor(ResultProcessor):

    def __init__(self, key: str):
        self._key = key

    def process(self, events_spark, process_event):
        events_df = events_spark.toPandas()

        def process_event(event):
            return {'upload_to_s3': None,
                    'media_filepath': f'{self._key}/processed-events/20200121_094518/{event["media_filename"]}',
                    'check_event_media': 'failed_media_downloaded',
                    'media_processing_begin': '2020-01-21 09:45:25.572256',
                    'media_processing_duration': '1.289841', 'media_processing_end': '2020-01-21 09:45:26.862097'}

        events_df['event_media_check_response'] = events_df.apply(lambda x: process_event(x), axis=1)
        return events_df


@mock_s3
def test_ds_format(spark_session: SparkSession):
    spark = spark_session

    project_name = '20200121-ds-format-test'
    project_bucket = 'nauto-test-ai'
    project_key = 'work/someuser'
    project_output_data_key = 'work/someuser/data/processed/{}'.format(project_name)

    boto3.client('s3').create_bucket(Bucket=project_bucket)

    formatter = DSFormatter(project_name, project_bucket, project_key, spark, in_qubole=False,
                            data_provider=StaticDataProvider(spark),
                            result_processor=TestResultProcessor(project_output_data_key))

    query = """
        SELECT s.*,
                  v.vin AS vin,
                  regexp_replace(regexp_replace(trim(lower(v.make)), ' ', '_'), '_&_', '_and_') AS make,
                  regexp_replace(regexp_replace(trim(lower(v.model)), ' ', '_'), '_&_', '_and_') AS model,
                  v.year
             FROM drt.judgements AS j
        LEFT JOIN drt.events AS e
               ON j.event_id = e.id
        LEFT JOIN device.severe_g AS s
               ON LOWER(HEX(e.device_id)) || '-' || LOWER(HEX(e.message_id)) = s.device_id || '-' || s.message_id
        LEFT JOIN dimension.vehicle AS v
               ON s.vehicle_id = v.vehicle_id
            WHERE j.created_at >= '2020-01-01'
              AND j.task_type = 'crashnet'
              AND (get_json_object(j.info, '$.near-collision-subquestion') like '%["pedestrian"]%'
                   OR get_json_object(j.info, '$.risky-manuever-subquestion') like '%["pedestrian"]%'
                   OR get_json_object(j.info, '$.what-did-hit') like '%["pedestrian"]%')
              AND s.message_type IN ('crashnet', 'severe-g-event')
    """
    events = DRTEvents(query, StaticDataProvider(spark))
    config = RequestConfig(prepare_new_request=True,
                           request_description='This is test dataset generation',
                           requested_by='john_smith@nauto.com',
                           prepared_by='foo_bar@nauto.com',
                           dataset_name='test_dataset_with_mock_data')
    formatter.generate_data_set(config, events)


@mock_s3
def test_ds_format_from_event_packager(spark_session: SparkSession):
    class EPStaticDataProvider(StaticDataProvider):
        def __init__(self, spark):
            super().__init__(spark)

        def get_events(self, query: str) -> pyspark.sql.dataframe.DataFrame:
            pickle = self._read_from_pickle('packaged_events')
            return self._spark.createDataFrame(pickle)

    spark = spark_session

    project_name = '20200121-ds-format-test-event-packager'
    project_bucket = 'nauto-test-ai'
    project_key = 'work/someuser'
    project_output_data_key = 'work/someuser/data/processed/{}'.format(project_name)

    boto3.client('s3').create_bucket(Bucket=project_bucket)

    formatter = DSFormatter(project_name, project_bucket, project_key, spark, in_qubole=False,
                            data_provider=EPStaticDataProvider(spark),
                            result_processor=TestResultProcessor(project_output_data_key))

    query = """
    SELECT pe.id, pe.type, pe.device_id, pe.message_id, pe.message_ts, pe.fleet_id, pe.s2_cell_id, 
            pe.accepted_at, pe.received_at, pe.created_at, pe.params,
                 v.vin AS vin,
                  regexp_replace(regexp_replace(trim(lower(v.make)), ' ', '_'), '_&_', '_and_') AS make,
                  regexp_replace(regexp_replace(trim(lower(v.model)), ' ', '_'), '_&_', '_and_') AS model,
                  v.year
             FROM event_packager.packaged_events AS pe
        LEFT JOIN dimension.vehicle AS v
               ON get_json_object(pe.params, '$.vehicle_id') = v.vehicle_id
            WHERE pe.message_day >= '2020-01-01' AND pe.message_day <= '2020-01-20'
              AND pe.type IN ('crashnet', 'severe-g-event')
              AND (
                      (pe.near_collision_subquestion.confidence=1 AND pe.near_collision_subquestion.value like '%["pedestrian"]%')
                  OR  (pe.risky_maneuver_subquestion.confidence=1 AND pe.risky_maneuver_subquestion.value like '%["pedestrian"]%')
                  OR  (pe.what_did_hit.confidence=1 AND pe.what_did_hit.value like '%["pedestrian"]%')
                  )
    """
    events = PackagedEvents(query, EPStaticDataProvider(spark))
    config = RequestConfig(prepare_new_request=True,
                           request_description='This is test dataset generation, from packaged events',
                           requested_by='john_smith@nauto.com',
                           prepared_by='foo_bar@nauto.com',
                           dataset_name='test_dataset_with_mock_packaged_events_data')
    formatter.generate_data_set(config, events)


@mock_s3
def test_input_data_without_media_in_params(spark_session):
    project_name = '20200121-ds-format-no-video'
    project_bucket = 'nauto-test-ai'
    project_key = 'work/someuser'
    project_output_data_key = 'work/someuser/data/processed/{}'.format(project_name)

    boto3.client('s3').create_bucket(Bucket=project_bucket)

    class NoMediaDataProvider(StaticDataProvider):
        def get_events(self, query: str) -> pyspark.sql.dataframe.DataFrame:
            result = super().get_events(query).toPandas()

            def remove_video(params):
                params = json.loads(params)
                del params[VIDEO_IN]
                del params[VIDEO_OUT]
                del params[SENSOR_FILES]
                return json.dumps(params)

            result['message_params'] = result.message_params.apply(remove_video)
            result = result.drop(columns=[VIDEO_IN, VIDEO_OUT, SENSOR_FILES])
            return self._spark.createDataFrame(result)

    formatter = DSFormatter(project_name, project_bucket, project_key, spark_session, in_qubole=False,
                            data_provider=NoMediaDataProvider(spark_session),
                            result_processor=TestResultProcessor(project_output_data_key))

    events = DRTEvents("mock query", NoMediaDataProvider(spark_session))
    config = RequestConfig(prepare_new_request=True,
                           request_description='This is test dataset generation',
                           requested_by='john_smith@nauto.com',
                           prepared_by='foo_bar@nauto.com',
                           dataset_name='test_dataset_with_mock_data')
    formatter.generate_data_set(config, events)


@mock_s3
def test_input_data_from_event_report(spark_session):
    project_name = '20200121-ds-format-event-report'
    project_bucket = 'nauto-test-ai'
    project_key = 'work/someuser'
    project_output_data_key = 'work/someuser/data/processed/{}'.format(project_name)

    boto3.client('s3').create_bucket(Bucket=project_bucket)

    class EventReportDataProvider(StaticDataProvider):

        def get_events(self, query: str) -> pyspark.sql.dataframe.DataFrame:
            pickle = self._read_from_pickle('event_report')
            return self._spark.createDataFrame(pickle)

    formatter = DSFormatter(project_name, project_bucket, project_key, spark_session, in_qubole=False,
                            data_provider=EventReportDataProvider(spark_session),
                            result_processor=TestResultProcessor(project_output_data_key))

    events = DRTEvents("event_reports query", EventReportDataProvider(spark_session))
    config = RequestConfig(prepare_new_request=True,
                           request_description='This is test dataset generation',
                           requested_by='john_smith@nauto.com',
                           prepared_by='foo_bar@nauto.com',
                           dataset_name='test_dataset_with_mock_data')
    formatter.generate_data_set(config, events)
