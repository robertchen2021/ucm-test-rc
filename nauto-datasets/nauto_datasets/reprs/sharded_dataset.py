import abc
import asyncio
import functools
import gc
import logging
import os
import tempfile
from pathlib import Path
from typing import (Any, Awaitable, Callable, Dict, Iterator, List, NamedTuple,
                    Optional, Tuple, TypeVar, Union)

from pyspark import Row
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from nauto_datasets.core import spark
from nauto_datasets.core.dataset import DatasetInstance
from nauto_datasets.core.serialization import FileLocation, FileSource
from nauto_datasets.core.spark import ParallelismConfig
from nauto_datasets.utils.boto import AsyncBotoS3Client
from nauto_datasets.utils.category import DummyMonoid, Monoid
from nauto_datasets.utils.tuples import NamedTupleMetaEx
from nauto_datasets.protos import shards_pb2


class ShardsSplitInfo(NamedTuple):
    shard_locations: List[FileLocation]
    examples_count: int
    not_included_count: int

    def to_pb(self) -> shards_pb2.ShardsSplitInfo:
        return shards_pb2.ShardsSplitInfo(
            shard_locations=[loc.to_pb() for loc in self.shard_locations],
            examples_count=self.examples_count,
            not_included_count=self.not_included_count)

    @staticmethod
    def from_pb(
            si_pb: shards_pb2.ShardsSplitInfo
    ) -> 'ShardsSplitInfo':
        return ShardsSplitInfo(
            shard_locations=[
                FileLocation.from_pb(loc_pb)
                for loc_pb in si_pb.shard_locations
            ],
            examples_count=si_pb.examples_count,
            not_included_count=si_pb.not_included_count)

    @staticmethod
    def pb_message_type() -> type:
        """Returns the type of the associate protobuf `Message`"""
        return shards_pb2.ShardsSplitInfo


class ShardedDatasetInfo(NamedTuple):
    splits: Dict[str, ShardsSplitInfo]
    split_by_column: Optional[str]

    def to_pb(self) -> shards_pb2.ShardedDatasetInfo:
        return shards_pb2.ShardedDatasetInfo(
            splits={
                split_name: split_info.to_pb() for split_name, split_info
                in self.splits.items()
            },
            split_by_column=self.split_by_column)

    @staticmethod
    def from_pb(
            sd_pb: shards_pb2.ShardedDatasetInfo
    ) -> 'ShardedDatasetInfo':
        split_by_column = None if sd_pb.split_by_column == ''\
            else sd_pb.split_by_column
        return ShardedDatasetInfo(
            split_by_column=split_by_column,
            splits={
                split_name: ShardsSplitInfo.from_pb(s_pb)
                for split_name, s_pb in sd_pb.splits.items()
            })

    @staticmethod
    def pb_message_type() -> type:
        """Returns the type of the associate protobuf `Message`"""
        return shards_pb2.ShardedDatasetInfo


class ShardProducer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def write(self,
              column_data: Dict[str, Any],
              s3_data: Dict[str, Union[bytes, List[bytes]]]) -> None:
        """Raises if data was not accepted and written to a shard file"""

    @abc.abstractmethod
    def finish(self) -> Optional[Tuple[List[Path], Monoid]]:
        """Finishes writing and returns paths to created files
        along with aggregation results.
        If result is None, then no data file has been created
        """


class ShardSpec(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def fetch_columns(self) -> List[str]:
        """
        Returns: names of the columns with lists of paths to
            s3 files. The data for these columns will be downloaded
            and provided to the associated `ShardProducer`
        """

    @abc.abstractmethod
    def get_producer(
            self,
            target_file_directory: Path,
            partition_id: int,
            batch_id: int
    ) -> ShardProducer:
        """Returns a `ShardProducer` writing the results to a local file

        Args:
            target_file_path: local path to the produced file
            partition_id: id of the dataset partition for which this producer
                is created
            batch_id: id of the batch within the partition

        Both `partition_id` and `batch_id` might be used to create a unique
        filename returned

        Returns: `ShardProducer`
        """

    @property
    def aggregation_monoid(self) -> type:
        """Overwrite this method to provide a custom `Monoid` type to
        to aggregate different results across the entire dataset """
        return DummyMonoid


class ShardJobResults(Monoid, metaclass=NamedTupleMetaEx):
    """Represents results of the single shard creation job.

    Attributes:
        total_examples: the total number of rows processed by the job
        failed_examples: the total number of rows for which encoding failed
            and consequently are not a part of the produced dataset.
        shard_locations: locations of produced shards
    """
    total_examples: int
    failed_examples: int
    shard_locations: List[FileLocation]

    @staticmethod
    def zero() -> 'ShardJobResults':
        return ShardJobResults(0, 0, [])

    @staticmethod
    def add(
            first: 'ShardJobResults',
            second: 'ShardJobResults'
    ) -> 'ShardJobResults':
        return ShardJobResults(
            first.total_examples + second.total_examples,
            first.failed_examples + second.failed_examples,
            first.shard_locations + second.shard_locations)


A = TypeVar('A')
B = TypeVar('B')


def map_batched_iterator(
        rows_iterator: Iterator[B],
        batch_size: Optional[int],
        map_fn: Callable[[B], A]
) -> Iterator[Tuple[int, List[A]]]:
    ind = 0
    batch_ind = 0
    batch_results = []
    for row in rows_iterator:
        batch_results.append(map_fn(row))
        if batch_size is not None and ind != 0 and (ind + 1) % batch_size == 0:
            yield (batch_ind, batch_results)
            batch_results = []
            batch_ind += 1
        ind += 1

    if len(batch_results) > 0:
        yield (batch_ind, batch_results)


def get_s3_files_fetcher(
        fetch_s3_data_columns: List[str],
        client: AsyncBotoS3Client
) -> Callable[[Row], Awaitable[Optional[Tuple[Dict[str, Any], Dict[str, List[bytes]]]]]]:

    async def get_media(row: Row) -> Optional[Tuple[Row,
                                                    Optional[Dict[str, List[bytes]]]]]:
        data_coros = []  # list of coroutines downloading s3 data
        for column_name in fetch_s3_data_columns:
            links = row[column_name]
            if links is None:
                # when data is missing - return None as a failure
                return row, None
            elif isinstance(links, str):
                data_coros.append(client.read_file(Path(links)))
            elif None not in links:
                data_coros.append(
                    client.read_file_list(list(map(Path, links))))
            else:
                return row, None
        try:
            s3_data = await asyncio.gather(*data_coros)
            return (row.asDict(), dict(zip(fetch_s3_data_columns, s3_data)))
        except Exception:
            logging.error(f'Could not get data from s3 for row {row}')
            return row, None

    return get_media


def data_to_shards(
        sess: SparkSession,
        data_df: DataFrame,
        target_location: FileLocation,
        shard_spec: ShardSpec,
        examples_per_shard: Optional[int] = None,
        par_config: ParallelismConfig = ParallelismConfig(keep_partitions=True),
        aio_boto_s3_client_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[ShardJobResults, Monoid]:
    """Creates dataset representation as sets of different files

    Each dataset split will correspond to a separate directory with shards

    Args:
        sess: spark session
        data_df: dataframe with rows grouped and transformmed to different shards.
            `data_df` should contain all the columns requested by the
            `shard_spec.fetch_columns`
        target_location: the root location of the resulting shard files
        shard_spec: used to produce each shard file
        examples_per_shard: the effective batch size of the processing job and the
            upper bound on the number of examples in each produced shard
        par_config: parallelism config governing the number of parallel jobs producing
            the dataset
    Returns:
        a dictionary mapping the split name to the job results with paths to associated
        shard files
    """
    agg_monoid: Monoid = shard_spec.aggregation_monoid
    aio_boto_s3_client_kwargs = aio_boto_s3_client_kwargs or {}

    def write_shards(
            partition_index: int,
            row_iterator: Iterator[Row]) -> Iterator[ShardJobResults]:
        if target_location.file_source == FileSource.LOCAL:
            # write to the destination directly
            local_dir = target_location.path
        elif target_location.file_source == FileSource.S3:
            # write to a temporary file on disk, before sending to s3
            local_dir = Path(
                tempfile.mkdtemp(f'part_{partition_index}'))
        else:
            raise ValueError(f'target_file source is not supported {target_location}')
        local_dir.mkdir(parents=True, exist_ok=True)

        async def write_shards_asynchronously(
                loop: asyncio.AbstractEventLoop) -> Tuple[ShardJobResults, Monoid]:

            agg_results: Monoid = agg_monoid.zero()
            shard_results = ShardJobResults.zero()

            async with AsyncBotoS3Client(loop, **aio_boto_s3_client_kwargs) as client:
                fetcher = get_s3_files_fetcher(
                    shard_spec.fetch_columns, client)

                for batch_ind, jobs_batch in map_batched_iterator(
                        row_iterator, examples_per_shard, fetcher):
                    gc.collect()

                    total_examples = len(jobs_batch)
                    failed_examples = 0
                    shard_locations = []

                    shard_producer = shard_spec.get_producer(
                        local_dir, partition_index, batch_ind)

                    # process rows in the order of their arrival
                    for row_job in asyncio.as_completed(jobs_batch, loop=loop):
                        column_data, s3_data = await row_job
                        try:
                            shard_producer.write(column_data, s3_data)
                        except Exception:
                            logging.exception('Shard producer could not write data.')
                            failed_examples += 1

                    maybe_shard_results = shard_producer.finish()
                    if maybe_shard_results is not None and len(maybe_shard_results[0]) > 0:
                        local_shards_paths, shard_info = maybe_shard_results

                        if target_location.file_source == FileSource.LOCAL:
                            shard_locations = [
                                FileLocation(path, FileSource.LOCAL)
                                for path in local_shards_paths
                            ]
                        elif target_location.file_source == FileSource.S3:
                            # send results over s3
                            destiny_locations = [
                                target_location.with_suffix(path.name)
                                for path in local_shards_paths
                            ]
                            try:
                                logging.info(
                                    f'Uploading a local shards: {local_shards_paths} '
                                    f'to {destiny_locations}')
                                await asyncio.wait(
                                    [
                                        client.upload_file(path, location.path)
                                        for path, location
                                        in zip(local_shards_paths, destiny_locations)
                                    ])
                                shard_locations = destiny_locations
                            except Exception:
                                logging.exception(
                                    f'Uploading a shard failed. '
                                    f'From {local_shards_paths} to {destiny_locations}')
                                failed_examples = len(jobs_batch)
                                shard_info = agg_monoid.zero()
                            # remove the local temporary files
                            for path in local_shards_paths:
                                os.remove(str(path))
                    else:
                        shard_info = agg_monoid.zero()
                        failed_examples = len(jobs_batch)

                    shard_results = ShardJobResults.add(
                        shard_results,
                        ShardJobResults(total_examples=total_examples,
                                        failed_examples=failed_examples,
                                        shard_locations=shard_locations))
                    agg_results = agg_results.add(agg_results, shard_info)

            return shard_results, agg_results

        loop = asyncio.get_event_loop()
        yield loop.run_until_complete(write_shards_asynchronously(loop))

    return par_config.repartition(sess, data_df) \
                     .rdd \
                     .mapPartitionsWithIndex(write_shards) \
                     .cache() \
                     .fold((ShardJobResults.zero(), agg_monoid.zero()),
                           lambda p1, p2: (ShardJobResults.add(p1[0], p2[0]),
                                           agg_monoid.add(p1[1], p2[1])))


def create_sharded_dataset(
        sess: SparkSession,
        di: DatasetInstance,
        target_location: FileLocation,
        shard_spec: ShardSpec,
        split_by_column: Optional[str] = None,
        reshuffle: bool = True,
        examples_per_shard: Optional[int] = None,
        par_config: ParallelismConfig = ParallelismConfig(keep_partitions=True),
        aio_boto_s3_client_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[ShardedDatasetInfo, Dict[str, Monoid]]:
        """Creates dataset consisting of multiple shards for a given dataset instance

        Each dataset split will correspond to a separate directory with different shards

        Args:
            sess: spark session
            di: dataset instance to transform
            target_location: the root location of the resulting dataset
            shard_spec: used to produce the files for the acquired data
            split_by_column: if provided then each split directory will be additionaly
                 divided into two subdirectories with names `<split_by_column>=True`
                 and `<split_by_column>=False`
            reshuffle: whether the data associated with each split should be reshuffled
            examples_per_shard: the effective batch size of the processing job and the
                upper bound on the number of examples in each produced shard
            par_config: parallelism config governing the number of parallel jobs producing
                the dataset
            aio_boto_s3_client_kwargs: additional arguments passed to aiobotocore's
                `session.create_client`. Used mostly for tests.

        Returns:
            sharded_dataset_info: ShardedDatasetInfo
            aggregation_results: an aggregation result for each dataset split
        """
        splits_dfs = di.load(sess)

        data_writer_fn = functools.partial(
            data_to_shards,
            sess=sess,
            shard_spec=shard_spec,
            examples_per_shard=examples_per_shard,
            par_config=par_config,
            aio_boto_s3_client_kwargs=aio_boto_s3_client_kwargs)

        splits: Dict[str, ShardsSplitInfo] = {}
        agg_infos: Dict[str, Monoid] = {}

        for split_name, split_df in splits_dfs.items():
            split_location = target_location.with_suffix(split_name)
            if split_by_column:
                pos_split = split_df.where(F.col(split_by_column))
                neg_split = split_df.where(~F.col(split_by_column))
                if reshuffle:
                    pos_split = spark.shuffle(pos_split)
                    neg_split = spark.shuffle(neg_split)

                pos_split_location = split_location.with_suffix(
                    f'{split_by_column}=True')
                neg_split_location = split_location.with_suffix(
                    f'{split_by_column}=False')

                pos_results, pos_agg = data_writer_fn(
                    data_df=pos_split, target_location=pos_split_location)
                neg_results, neg_agg = data_writer_fn(
                    data_df=neg_split, target_location=neg_split_location)
                results = ShardJobResults.add(pos_results, neg_results)
                agg = shard_spec.aggregation_monoid.add(pos_agg, neg_agg)
            else:
                if reshuffle:
                    split_df = spark.shuffle(split_df)
                results, agg = data_writer_fn(
                    data_df=split_df, target_location=split_location)

            splits[split_name] = ShardsSplitInfo(
                shard_locations=results.shard_locations,
                examples_count=results.total_examples - results.failed_examples,
                not_included_count=results.failed_examples)
            agg_infos[split_name] = agg

        return (
            ShardedDatasetInfo(splits=splits, split_by_column=split_by_column),
            agg_infos
        )
