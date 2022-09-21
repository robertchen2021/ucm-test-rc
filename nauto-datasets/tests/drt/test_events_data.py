from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession

from nauto_datasets.drt.reprs.ds_data import PackagedEvents, DRTEvents


def test_packaged_events_data(spark_session: SparkSession):
    packaged_events_path = str(
        (Path(__file__).parent.parent / 'test_data' / 'ds_data' / 'packaged_events.pkl').resolve())
    packaged_events = PackagedEvents("mock_query", spark=spark_session,
                                     data_frame=pd.read_pickle(packaged_events_path).drop(columns=['make', 'model']))

    packaged_events_df = packaged_events.get()
    assert 'videos' in packaged_events_df
    assert 'sensors' in packaged_events_df


def test_event_report_data(spark_session: SparkSession):
    event_report_path = str((Path(__file__).parent.parent / 'test_data' / 'ds_data' / 'event_report.pkl').resolve())
    event_report = DRTEvents("mock_query", spark=spark_session, data_frame=pd.read_pickle(event_report_path))

    event_report_df = event_report.get()
    assert 'videos' in event_report_df
    assert 'sensors' in event_report_df
