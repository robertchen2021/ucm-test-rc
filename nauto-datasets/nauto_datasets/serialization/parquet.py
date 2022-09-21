from typing import List, Optional, Any

from pyspark.sql import DataFrame, SparkSession, functions as F

from nauto_datasets.core.schema import RecordSchema
from nauto_datasets.core.serialization import (DataframeSerializer,
                                               FileLocation, SerializedData)

from nauto_datasets.protos import serialization_pb2 as ser_pb


class ParquetData(SerializedData):
    def __init__(self,
                 location: FileLocation,
                 compression: Optional[str],
                 partitioning_columns: Optional[List[str]] = None) -> None:
        """Creates a description of data serialized as a parquet file

        Args:
            location: location of the parquet file. Might point to a directory
                tree with several parquet files which should be concatenated
                when reading
            compression: compression algorithm used, e.g. 'gzip', 'snappy' or
                defualt compression as specified by spark when None is provided
            partition_columns: the columns by which the data should be further
                split into directory tree
        """

        self._location = location
        if compression == '':
            self._compression = None
        else:
            self._compression = compression
        if partitioning_columns is not None and len(partitioning_columns) == 0:
            self._partitioning_columns = None
        else:
            self._partitioning_columns = partitioning_columns

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'location={self._location},\n'
                f'compression={self._compression}\n'
                f'partitioning={self._partitioning_columns})')

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def location(self) -> FileLocation:
        return self._location

    @property
    def compression(self) -> Optional[str]:
        return self._compression

    @property
    def partitioning_columns(self) -> Optional[List[str]]:
        return self._partitioning_columns

    def read(self, sess: SparkSession, schema: RecordSchema) -> DataFrame:
        """Reads resources serialized as parquet files under a common
        tree root node specified by `location`

        Args:
            sess: spark session to use for reading
            schema: schema describing each record to be read
        Returns:
            a data frame with rows conforming to the `schema`
        """
        path = self.location.to_url()
        com_schema = schema.combined_schema()
        data_df = sess.read.schema(com_schema).parquet(path)

        return data_df

    def to_pb(self) -> ser_pb.ParquetData:
        """Serializes this `ParquetData` as protobuf `Message`"""
        return ser_pb.ParquetData(
            location=self._location.to_pb(),
            compression=self._compression,
            partitioning_columns=self._partitioning_columns)

    @staticmethod
    def from_pb(pd_pb: ser_pb.ParquetData) -> 'ParquetData':
        """Deserializes `ParquetData` from protobuf `Message`"""
        return ParquetData(
            location=FileLocation.from_pb(pd_pb.location),
            compression=pd_pb.compression,
            partitioning_columns=pd_pb.partitioning_columns
        )

    @staticmethod
    def pb_message_type() -> type:
        """Returns the type of the associate protobuf `Message`"""
        return ser_pb.ParquetData


class ParquetSerializer(DataframeSerializer):
    def __init__(self,
                 sess: SparkSession,
                 compression: str = 'gzip',
                 coalesce_to_one: bool = True) -> None:
        """Creates a `ParquetSerializer`
        Args:
            sess: Spark session
            compression: compression algorithm used, e.g. 'gzip', 'snappy' or
                defualt compression as specified by spark when None is provided
            coalesce_to_one: if data should be coalesced into one partition
                before saving
        """
        self._sess = sess
        self._compression = compression
        self._coalesce = coalesce_to_one

    def write(self,
              location: FileLocation,
              df: DataFrame,
              schema: RecordSchema,
              partition_columns: Optional[List[str]]) -> ParquetData:
        columns = [
            F.col(col_name) for col_name in schema.combined_schema().fieldNames()
        ]
        path = location.to_url()
        df = df.select(columns)
        if self._coalesce:
            df = df.coalesce(1)

        df.write.parquet(
            path=path,
            partitionBy=partition_columns,
            compression=self._compression)

        return ParquetData(
            location=location,
            compression=self._compression,
            partitioning_columns=partition_columns)
