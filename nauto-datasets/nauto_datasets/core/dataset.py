import abc
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional

from google.protobuf.timestamp_pb2 import Timestamp
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from nauto_datasets.core.schema import RecordSchema
from nauto_datasets.core.serialization import (DataframeSerializer,
                                               FileHandler, FileLocation,
                                               SerializedData)
from nauto_datasets.core.spark import ParallelismConfig
from nauto_datasets.core.splits import DataSplits, SplitStrategy
from nauto_datasets.protos import dataset_pb2


class DataSource(metaclass=abc.ABCMeta):
    """Abstract class describing a data source capable of producing
    the data used for the creation of datasets.
    """

    @abc.abstractmethod
    def produce(self,
                sess: SparkSession,
                since: Optional[datetime] = None,
                until: Optional[datetime] = None,
                **kwargs: Dict[str, Any]) -> DataFrame:
        """Returns data for specified time range.

        Args:
            since: if None, then the data will taken since the beginning
                of time.
            until: if None, then there will be no upperbound on the time
                of each returned row
            kwargs: additional keyword arguments

        Returns:
            a dataframe with dataset examples.
        """

    @abc.abstractmethod
    def record_schema(self) -> RecordSchema:
        """The schema desriring the rows produced by this `DataSource`"""
        pass

    @staticmethod
    def instance_to_pb(data_source: 'DataSource') -> dataset_pb2.DataSource:
        """Serializes a particular instance of `DataSource` as one of the
        variants of `dataset_pb2.DatasetSource`

        The following instances are accepted:
        - @{nauto_datasets.drt.data_source.DRTDataSource}

        Args:
            data_source: an instance of `DataSource`

        Returns:
            `DataSource` proto `Message` with the chosen instance as a
            variant.
        """
        from nauto_datasets.drt.data_source import DRTDataSource
        if isinstance(data_source, DRTDataSource):
            return dataset_pb2.DataSource(drt_data_source=data_source.to_pb())
        else:
            raise ValueError(f'Unsupported data_source type {data_source}')

    @staticmethod
    def instance_from_pb(data_source_pb: dataset_pb2.DataSource) -> 'DataSource':
        """Deserializes protobuf `Message` with a particular instance of `DataSource`

        This class handles the following instances:
        - @{nauto_datasets.drt.data_source.DRTDataSource}

        Args:
            data_source_pb: a `dataset_pb2.DataSource` probotuf `Message`

        Returns:
            `DataSource` instance
        """
        # introducing circural dependency for variant handling
        from nauto_datasets.drt.data_source import DRTDataSource
        variant_name = data_source_pb.WhichOneof('data_source')
        if variant_name == 'drt_data_source':
            return DRTDataSource.from_pb(data_source_pb.drt_data_source)
        else:
            raise ValueError(
                f'Unsupported data source type {variant_name}')


class DatasetDescription(NamedTuple):
    """The ultimate dataset description is in fact a pair of the
    DataSource responsible for producing the data and the
    SplitStrategy returning logical partitioning.
    """
    data_source: DataSource
    split_strategy: SplitStrategy

    def produce_dataset(
            self,
            sess: SparkSession,
            since: Optional[datetime] = None,
            until: Optional[datetime] = None,
            par_config: ParallelismConfig = ParallelismConfig(
                keep_partitions = True),
            **data_source_args: Dict[str, Any]
    ) -> DataSplits:
        """Creates a dataset set by calling the source of data with
        provided arguments and then applying splitting to the results.
        """
        data_df = self.data_source.produce(
            sess=sess, since=since, until=until, **data_source_args)

        data_df = par_config.repartition(sess, data_df)

        return self.split_strategy.split(data_df)

    def update_dataset(
            self,
            sess: SparkSession,
            old_splits: DataSplits,
            since: datetime,
            until: Optional[datetime] = None,
            par_config: ParallelismConfig = ParallelismConfig(
                keep_partitions = True),
            **data_source_args: Dict[str, Any]
    ) -> DataSplits:
        """Updates old dataset with data from the new time range

        The new data for the time range starting at `since` will
        be generated and later merged with the `old_splits` according
        to the implementation from the `self.split_strategy`
        """
        new_splits = self.produce_dataset(
            sess, since, until, par_config, **data_source_args)

        return self.split_strategy.merge(old_splits, new_splits)

    def to_pb(self) -> dataset_pb2.DatasetDescription:
        """Serializes description as a protobuf `Message`"""
        return dataset_pb2.DatasetDescription(
            data_source=DataSource.instance_to_pb(self.data_source),
            split_strategy=SplitStrategy.instance_to_pb(self.split_strategy))

    @staticmethod
    def from_pb(dd_pb: dataset_pb2.DatasetDescription) -> 'DatasetDescription':
        """Deserializes description from a protobuf `Message`"""
        return DatasetDescription(
            data_source=DataSource.instance_from_pb(
                dd_pb.data_source),
            split_strategy=SplitStrategy.instance_from_pb(
                dd_pb.split_strategy))

    @staticmethod
    def pb_message_type() -> type:
        return dataset_pb2.DatasetDescription


class DatasetInstance(NamedTuple):
    """A meta description of the dataset

    Attributes:
        name: name of the dataset
        creation_time: the time dataset was materialized
        data_since: the lower bound on time of the rows
            in the dataset
        data_until: the upper bound on time of the rows in
            the dataset
        schema: schema representing each row of this dataset
        splits: points to the data representing each logical
            split, e.g. 'training', 'validation', 'test'
    """
    name: str
    creation_time: datetime
    data_since: Optional[datetime]
    data_until: datetime
    schema: RecordSchema
    splits: Dict[str, SerializedData]

    def load(self, sess: SparkSession) -> DataSplits:
        return {
            name: ser_data.read(sess, self.schema)
            for name, ser_data in self.splits.items()
        }

    @staticmethod
    def create(
            sess: SparkSession,
            name: str,
            location: FileLocation,
            dataset_desc: DatasetDescription,
            since: Optional[datetime],
            until: Optional[datetime],
            data_serializer: DataframeSerializer,
            partition_columns: Optional[List[str]] = None,
            par_config: ParallelismConfig = ParallelismConfig(
                keep_partitions = True),
            **data_source_args: Dict[str, Any]
    ) -> 'DatasetInstance':
        """Creates a dataset instance.

        The dataset is produced and the serialized under provided location

        Args:
            sess: spark session
            name: name of the resulting dataset
            location: a physical location where the data should be saved
            dataset_desc: description of the dataset
            since: the lower bound on time of the rows
                in the dataset
            until: the upper bound on time of the rows in
                the dataset
            data_serializer: an instance of `DataframeSerializer` used
                to write produced split to files under `location`
            partition_columns: the columns, by which each split should
                by additionally partitioning into nested directories,
                e.g. <location>/train/column_ex_1=True/column_ex_2='SomeName'/<data_file>
            par_config: a parallelism config used to produce the dataset
            data_source_args: a keyword arguments passed further to the
               `DataSource``
        """
        creation_time = datetime.now()
        if until is None:
            until = creation_time

        data_splits = dataset_desc.produce_dataset(
            sess, since=since, until=until, par_config=par_config,
            **data_source_args)

        schema = dataset_desc.data_source.record_schema()

        splits = {
            split_name: data_serializer.write(
                location=location.with_suffix(split_name),
                df=split_df,
                schema=schema,
                partition_columns=partition_columns)
            for split_name, split_df in data_splits.items()
        }

        return DatasetInstance(
            name,
            creation_time=creation_time,
            data_since=since,
            data_until=until,
            schema=schema,
            splits=splits)

    @staticmethod
    def update(
        sess: SparkSession,
        old_di: 'DatasetInstance',
        name: str,
        location: FileLocation,
        dataset_desc: DatasetDescription,
        until: Optional[datetime],
        data_serializer: DataframeSerializer,
        partition_columns: Optional[List[str]] = None,
        par_config: ParallelismConfig = ParallelismConfig(
            keep_partitions = True),
        **data_source_args: Dict[str, Any]
    ) -> 'DatasetInstance':
        """Updates old DatasetInstance with newer data

        The new dataset instance is produced and the serialized under provided
        location

        Args:
            sess: spark session
            old_di: original DatasetInstance which needs to be updasted
            name: name of the resulting dataset
            location: a physical location where the updated dataset should be saved
            dataset_desc: description of the dataset
            until: the upper bound on time of the rows in
                the dataset
            data_serializer: an instance of `DataframeSerializer` used
                to write produced split to files under `location`
            partition_columns: the columns, by which each split should
                by additionally partitioning into nested directories,
                e.g. <location>/train/column_ex_1=True/column_ex_2='SomeName'/<data_file>
            par_config: a parallelism config used to produce the dataset
            data_source_args: a keyword arguments passed further to the
               `DataSource``

        Returns:
            Updated DatasetInstance
        """
        old_splits = old_di.load(sess)

        creation_time = datetime.now()
        until = until or creation_time

        new_splits = dataset_desc.update_dataset(
            sess,
            old_splits,
            since=old_di.data_until,
            until=until,
            par_config=par_config,
            **data_source_args)

        splits = {
            split_name: data_serializer.write(
                location=location.with_suffix(split_name),
                df=split_df,
                schema=old_di.schema,
                partition_columns=partition_columns)
            for split_name, split_df in new_splits.items()
        }

        return DatasetInstance(
            name,
            creation_time=creation_time,
            data_since=old_di.data_since,
            data_until=until,
            schema=old_di.schema,
            splits=splits)

    def to_pb(self) -> dataset_pb2.DatasetInstance:
        """Serializes `DatasetInstance` as protobuf `Message`"""
        creation_time_ts = Timestamp()
        creation_time_ts.FromDatetime(self.creation_time)

        data_since_ts = Timestamp()
        if self.data_since is not None:
            data_since_ts.FromDatetime(self.data_since)

        data_until_ts = Timestamp()
        data_until_ts.FromDatetime(self.data_until)

        return dataset_pb2.DatasetInstance(
            name=self.name,
            creation_time=creation_time_ts,
            data_since=data_since_ts,
            data_until=data_until_ts,
            schema=self.schema.to_pb(),
            splits={
                name: SerializedData.instance_to_pb(ser_data)
                for name, ser_data in self.splits.items()
            }
        )

    @staticmethod
    def from_pb(di_pb: dataset_pb2.DatasetInstance) -> 'DatasetInstance':
        """Deserializes instance from a protobuf `Message`"""
        return DatasetInstance(
            name=di_pb.name,
            creation_time=di_pb.creation_time.ToDatetime(),
            data_since=di_pb.data_since.ToDatetime(),
            data_until=di_pb.data_until.ToDatetime(),
            schema=RecordSchema.from_pb(di_pb.schema),
            splits={
                name: SerializedData.instance_from_pb(ser_data)
                for name, ser_data in di_pb.splits.items()
            }
        )

    @staticmethod
    def pb_message_type() -> type:
        return dataset_pb2.DatasetInstance


class DatasetRepresentation(metaclass=abc.ABCMeta):
    """An interface describing arbitrary representations or views
    of datasets.

    `DatasetInstance` is essentially a partitioning of high level
    data into logical sets (train, test and so on).

    There might be different, but more concrete representations of
    a particular dataset, e.g. each row of the original `DatasetInstance`
    might be transformed to a separate json/csv file, or there might
    be many tfrecords extracted from every dataset split.
    """

    @abc.abstractmethod
    def create(self,
               sess: SparkSession,
               di: DatasetInstance,
               **args: Dict[str, Any]) -> Any:
        """
        Creates a dataset representation.
        Args:
            sess: spark session
            di: dataset instance to transform
            args: additional keyword arguments
        """
