import abc
from typing import Optional

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import MapType, StringType

from nauto_datasets.drt.reprs.ds_data import SENSORS


class ResultProcessor(abc.ABC):
    COLUMNS = ['device_id', 'event_id', 'message_id', 'message_ts', 'event_message_type', 'event_message_params',
               'utc_basetime', 'media_filename']

    @abc.abstractmethod
    def process(self, events_spark, process_event) -> pd.DataFrame:
        pass


class PandasResultProcessor(ResultProcessor):
    def process(self, events_spark, process_event) -> pd.DataFrame:
        process_results = events_spark.toPandas()
        process_results['event_media_check_response'] = process_results.apply(lambda x: process_event(x), axis=1)
        process_results = process_results[self.COLUMNS + ['event_media_check_response']]
        return process_results


class SparkResultProcessor(ResultProcessor):

    def __init__(self, spark, partition_count: Optional[int] = 200):
        self._spark = spark
        self._partition_count = partition_count

    def process(self, events_spark, process_event) -> pd.DataFrame:
        process_event_udf = F.udf(process_event, MapType(StringType(), StringType()))
        events_spark = (
            self._spark.createDataFrame(events_spark.toPandas().loc[:, self.COLUMNS + ['videos', SENSORS]]))

        process_results = (events_spark
                           .repartition(self._partition_count)
                           .withColumn('event_media_check_response',
                                       process_event_udf(F.struct([events_spark[x] for x in events_spark.columns])))
                           .select(self.COLUMNS + ['event_media_check_response'])
                           ).toPandas()
        return process_results
