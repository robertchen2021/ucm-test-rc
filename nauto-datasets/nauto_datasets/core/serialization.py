import abc
import asyncio
import shutil
from concurrent import futures
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from google.protobuf.message import Message
from pyspark.sql import DataFrame, SparkSession

from nauto_datasets.core.schema import RecordSchema
from nauto_datasets.protos import serialization_pb2 as ser_pb
from nauto_datasets.utils import protobuf
from nauto_datasets.utils.boto import (AsyncBotoS3Client, BotoS3Client,
                                       path_to_bucket_and_key)
from nauto_datasets.utils.tuples import NamedTupleMetaEx


class FileSource(Enum):
    LOCAL = 1
    HDFS = 2
    S3 = 3

    def to_pb(self) -> int:
        """Serializes `FileSource` as protobuf enum"""
        return self.value - 1

    @staticmethod
    def from_pb(fs_pb: int) -> 'FileSource':
        """Reads `FileSource` from protobuf enum"""
        return FileSource[ser_pb.FileSource.keys()[fs_pb]]

    @staticmethod
    def pb_message_type() -> type:
        """Returns the type of the associate protobuf `Message`"""
        return int


# NamedTupleMetaEx instead of NamedTuple to unpickle with
# methods
class FileLocation(metaclass=NamedTupleMetaEx):
    path: Path
    file_source: FileSource

    def to_url(self) -> str:
        """Returns url representing path, e.g. s3://<path>"""
        if self.file_source == FileSource.LOCAL:
            return str(self.path)

        path = str(self.path)
        path = '/' + path if path[0] != '/' else path
        if self.file_source == FileSource.HDFS:
            return 'hdfs:/' + path
        elif self.file_source == FileSource.S3:
            return 's3:/' + path
        else:
            raise ValueError('Invalid file_source')

    def with_suffix(self, suffix: Union[str, Path]) -> 'FileLocation':
        """Returns a new `FileLocaiton` with suffix path
        appended at the end of the `path`"""
        return self._replace(path=self.path / suffix)

    def to_pb(self) -> ser_pb.FileLocation:
        """Serializes the location as protobuf `Message`"""
        return ser_pb.FileLocation(
            path=str(self.path),
            file_source=self.file_source.to_pb()
        )

    @staticmethod
    def from_pb(fl_pb: ser_pb.FileLocation) -> 'FileLocation':
        """Serializes the location from protobuf `Message`"""
        return FileLocation(
            path=Path(fl_pb.path),
            file_source=FileSource.from_pb(fl_pb.file_source)
        )

    @staticmethod
    def pb_message_type() -> type:
        """Returns the type of the associate protobuf `Message`"""
        return ser_pb.FileLocation


class FileHandler:
    """Auxiliary class responsible for handling most common operations
    on files from different locations.
    """

    def __init__(self, s3_client: Optional[BotoS3Client] = None) -> None:
        self._s3_client = s3_client

    def _get_s3_client(self) -> BotoS3Client:
        if self._s3_client is None:
            self._s3_client = BotoS3Client()
        return self._s3_client

    def save_data(
            self,
            data: Union[str, bytes],
            location: FileLocation
    ) -> None:
        """Saves provided data under the given `location`."""
        if location.file_source == FileSource.LOCAL:
            mod = 'w' if isinstance(data, str) else 'wb'
            parent = location.path.parent
            if not parent.exists():
                parent.mkdir(parents=True)
            with open(location.path, mod) as f:
                f.write(data)
        elif location.file_source == FileSource.S3:
            s3_client = self._get_s3_client()
            s3_client.write_file(data, location.path)

        else:
            raise NotImplementedError(
                f'Not supported location {location.file_source}')

    def read_data(self, location: FileLocation) -> bytes:
        """Reads contents of the file under given `location`."""
        if location.file_source == FileSource.LOCAL:
            with open(location.path, 'rb') as f:
                return f.read()

        elif location.file_source == FileSource.S3:
            s3_client = self._get_s3_client()
            return s3_client.read_file(location.path)

        else:
            raise NotImplementedError(
                f'Not supported location {location.file_source}')

    def delete(self, location: FileLocation) -> None:
        """Delete contents of file or directory of given `location`."""
        if location.file_source == FileSource.LOCAL:
            if location.path.is_dir():
                shutil.rmtree(location.path)

            elif location.path.is_file():
                location.path.unlink()

        elif location.file_source == FileSource.S3:
            s3_client = self._get_s3_client()
            bucket_name, prefix = path_to_bucket_and_key(location.path)
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=bucket_name,
                                               Prefix=prefix)

            for page in page_iterator:
                delete_keys = [{'Key': obj['Key']} for obj in page['Contents']]
                paginated_delete_keys = {"Objects": delete_keys}
                s3_client.delete_objects(Bucket=bucket_name,
                                         Delete=paginated_delete_keys)

        else:
            raise NotImplementedError(
                f'Not supported location {location.file_source}')


class AsyncFileHandler:
    """Auxiliary class responsible for handling most common operations
    on files from different locations in an asynchronous fashion.
    """

    def __init__(self,
                 client: AsyncBotoS3Client,
                 thread_pool: futures.ThreadPoolExecutor) -> None:
        """Create `AsyncFileHandler`

        Args:
            client: asynchronous boto s3 client
            thread_pool: pool used to run operations on local
                files in asynchronous way.
        """
        self._client = client
        self._thread_pool = thread_pool

    async def save_data(
            self,
            data: Union[str, bytes],
            location: FileLocation
    ) -> None:
        """Saves provided data under the given `location`."""
        if location.file_source == FileSource.LOCAL:
            loop = asyncio.get_event_loop()

            def write_file():
                mod = 'w' if isinstance(data, str) else 'wb'
                parent = location.path.parent
                if not parent.exists():
                    parent.mkdir(parents=True)
                with open(location.path, mod) as f:
                    f.write(data)
            return await loop.run_in_executor(
                self._thread_pool, write_file)

        elif location.file_source == FileSource.S3:
            await self._client.write_file(data, location.path)

        else:
            raise NotImplementedError(
                f'Not supported location {location.file_source}')

    async def read_data(self, location: FileLocation) -> bytes:
        """Reads contents of the file under given `location`."""
        if location.file_source == FileSource.LOCAL:
            loop = asyncio.get_event_loop()

            def read_file():
                with open(location.path, 'rb') as f:
                    return f.read()
            return await loop.run_in_executor(
                self._thread_pool, read_file)

        elif location.file_source == FileSource.S3:
            return await self._client.read_file(location.path)

        else:
            raise NotImplementedError(
                f'Not supported location {location.file_source}')


class SerializedData(metaclass=abc.ABCMeta):
    """An abstract class describing serialized resources"""

    @abc.abstractmethod
    def read(self, sess: SparkSession, schema: RecordSchema) -> DataFrame:
        """Reads serialized resources.

        Args:
            sess: spark session to use for reading
            schema: schema describing each record to be read
        Returns:
            a data frame with rows conforming to the `schema`
        """

    @abc.abstractproperty
    def location(self) -> FileLocation:
        """Returns file location of the resource"""

    @staticmethod
    def instance_to_pb(ser_data: 'SerializedData') -> ser_pb.SerializedData:
        """Serializes this particular instance as one of the variants of
        `ser_pb.SerializedData`

        This class accepts the following instances:
        - @{nauto_datasets.serialization.parquet.ParquetData}

        Args:
            ser_data: an instance of `SerializedData`

        Returns:
            `SerializedData` proto `Message` with the chosen `SerializedData`
            instance as a variant.
        """
        # introducing circural dependency for variant handling
        from nauto_datasets.serialization import parquet
        if isinstance(ser_data, parquet.ParquetData):
            return ser_pb.SerializedData(parquet_data=ser_data.to_pb())
        else:
            raise ValueError(f'Unsupported ser_data type {ser_data}')

    @staticmethod
    def instance_from_pb(sd_pb: ser_pb.SerializedData) -> 'SerializedData':
        """Deserializes protobuf `Message` with a particular instance of `SerializedData`

        This class handles the following instances:
        - @{nauto_datasets.serialization.parquet.ParquetData}

        Args:
            sd_pb: a `serialization_pb2.SerializedData` protobuf `Message`

        Returns:
            `SerializedData` instance
        """
        # introducing circural dependency for variant handling
        from nauto_datasets.serialization import parquet
        variant_name = sd_pb.WhichOneof('serialized_data')
        if variant_name == 'parquet_data':
            return parquet.ParquetData.from_pb(sd_pb.parquet_data)
        else:
            raise ValueError(
                f'Unsupported serialized data type {variant_name}')


class DataframeSerializer(metaclass=abc.ABCMeta):
    """An interface defining a concrete instance of `SerializedData` writer."""

    @abc.abstractmethod
    def write(self,
              location: FileLocation,
              df: DataFrame,
              schema: RecordSchema,
              partition_columns: Optional[List[str]] = None) -> SerializedData:
        """Writes contents of dataframe to a file under provided location.

        Args:
            location: location where `df` should be saved
            df: data frame with the contents to serialize
            schema: schema describing the columns of the serialized data

        Returns:
            serialized_data: a description of the file with written contents.
        """


class ProtobufSerializable(metaclass=abc.ABCMeta):
    """Definition of the interface for the class which is serializable as
    protobuf `Message`.
    """

    @abc.abstractmethod
    def to_pb(self) -> Message:
        pass

    @staticmethod
    @abc.abstractmethod
    def from_pb(pb_msg: Message) -> 'ProtobufSerializable':
        pass

    @staticmethod
    def pb_message_type() -> type:
        """Returns the type of the associate protobuf `Message`"""


def save_as_proto_txt_file(
        proto_serializable: ProtobufSerializable,
        file_location: FileLocation,
        file_handler: FileHandler
) -> None:
    """Serializes `proto_serializable` as a textual representation
    of the relevant protobuf `Message` and writes it to a given
    file location.
    """
    dd_pb = proto_serializable.to_pb()
    dd_pbtxt = protobuf.message_to_txt(dd_pb)
    file_handler.save_data(dd_pbtxt, file_location)


def read_from_proto_txt_file(
        proto_serializable_class: type,
        file_location: FileLocation,
        file_handler: FileHandler
) -> ProtobufSerializable:
    """Deserializes `proto_serializable_class` from a textual representation
    of the relevant protobuf `Message` saved at `file_location`.
    """
    pbtxt = file_handler.read_data(file_location)
    pb = protobuf.parse_message_from_txt(
        proto_serializable_class.pb_message_type(), pbtxt)
    return proto_serializable_class.from_pb(pb)
