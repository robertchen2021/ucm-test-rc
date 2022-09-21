import numpy as np
from typing import Dict, NamedTuple, Optional

from pyspark.sql import DataFrame, DataFrameReader, SparkSession
from pyspark.sql import functions as F, types as spark_types
from nauto_datasets.utils.numpy import NDArray

JdbcConnection = DataFrameReader


def jdbc_connection(sess: SparkSession,
                    jdbc_url: str,
                    **options: Dict[str, str]
) -> JdbcConnection:
    """Returns a `JdbcConnection` for the given url and additional options"""
    return sess.read.format('jdbc').options(url=jdbc_url, **options)


def run_sql(jdbc_conn: JdbcConnection, sql: str) -> DataFrame:
    """Returns DataFrame with results of the provided `sql` query"""
    return jdbc_conn.options(dbtable=sql).load()


def shuffle(data_df: DataFrame) -> DataFrame:
    """Shuffles the order of the rows in `data_df`"""
    return data_df.orderBy(F.rand())


class ParallelismConfig(NamedTuple):
    """Expresses configuration determining the default partitioning
    of spark DataFrames.

    Proper partitioning enabling the most efficient parallel computations
    might depend on different factors. This class is merely a simple heuristic
    calculating the number of partitions based on the cluster parallelism
    level and additional factor provided by the user.

    Attributes:
        keep_partitions: if True no repartitioning should be performed
        mult: the multiplication factor applied to chosen `parallelism_level`
        parallelism_level: the number describing the parallelism in the
            cluster. If None, then spark's `defaultParallelism` is chosen
            by default.
    """
    keep_partitions: bool
    mult: int = 1
    parallelism_level: Optional[int] = None

    def new_partitions_count(self, sess: SparkSession) -> Optional[int]:
        """Returns the number of partitions to choose by default for a dataframe.
        If None, then the partitioning should be kept.
        """
        if self.keep_partitions:
            return None

        if self.parallelism_level is not None:
            return self.parallelism_level * self.mult
        else:
            return sess.sparkContext.defaultParallelism * self.mult

    def repartition(self, sess: SparkSession, df: DataFrame) -> DataFrame:
        """Potentially repartitions the `df` according to the configuration.

        Returns: repatitioned version of `df`
        """
        new_count = self.new_partitions_count(sess)
        if new_count is None:
            return df
        else:
            return df.repartition(new_count)


def spark_dt_to_numpy_dt(spark_dt: spark_types.DataType) -> type:
    """Converts an instance of pyspark's DataType to the numpy counterpart.

    In case there is no clear conversion, `np.object` is returned
    """
    if isinstance(spark_dt, spark_types.ByteType):
        return np.byte
    elif isinstance(spark_dt, spark_types.ShortType):
        return np.int16
    elif isinstance(spark_dt, spark_types.IntegerType):
        return np.int32
    elif isinstance(spark_dt, spark_types.LongType):
        return np.int64
    elif isinstance(spark_dt, spark_types.FloatType):
        return np.float32
    elif isinstance(spark_dt, spark_types.DoubleType):
        return np.float64
    elif isinstance(spark_dt, spark_types.BooleanType):
        return np.bool
    elif isinstance(spark_dt, spark_types.StringType):
        return np.str
    elif isinstance(spark_dt, spark_types.TimestampType):
        return np.datetime64
    elif isinstance(spark_dt, spark_types.ArrayType):
        return NDArray[spark_dt_to_numpy_dt(spark_dt.elementType)]
    else:
        raise ValueError(f'Unsupported spark type: {spark_dt}')
