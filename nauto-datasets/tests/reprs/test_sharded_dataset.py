import asyncio
import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

import aiobotocore
import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from nauto_datasets.core.serialization import (FileHandler, FileLocation,
                                               FileSource,
                                               read_from_proto_txt_file,
                                               save_as_proto_txt_file)
from nauto_datasets.core.spark import ParallelismConfig
from nauto_datasets.reprs import sharded_dataset
from nauto_datasets.utils.category import Monoid



class AllDataAggregator(Monoid):
    def __init__(self, s3_data: pd.DataFrame) -> None:
        self.s3_data = s3_data

    @staticmethod
    def zero() -> 'AllDataAggregator':
        return AllDataAggregator(pd.DataFrame())

    @staticmethod
    def add(a1: 'AllDataAggregator',
            a2: 'AllDataAggregator') -> 'AllDataAggregator':
        return AllDataAggregator(pd.concat([a1.s3_data, a2.s3_data]))


def get_json_file_name(part_ind: int, batch_ind: int, id: int) -> str:
    return f'{part_ind}_{batch_ind}_{id}.json'


class PickyJsonShardProducer(sharded_dataset.ShardProducer):
    """
    Saves rows with id % 3 == 0
    Raises on rows with id % 3 == 2 or 1
    """

    def __init__(self,
                 data_dir: Path,
                 part_ind: int,
                 batch_ind: int) -> None:
        self.data_dir = data_dir
        self.part_ind = part_ind
        self.batch_ind = batch_ind

        self.written_files = []
        self.accumulator = AllDataAggregator.zero()

    def write(self,
              column_data: Dict[str, Any],
              s3_data: Dict[str, Union[List[bytes], bytes]]) -> None:
        if column_data['id'] % 3 == 0:
            data = s3_data.copy()
            data['id'] = column_data['id']
            result = AllDataAggregator(
                pd.DataFrame(
                    [pd.Series(index=list(data), data=list(data.values()))]
                )
            )
            self.accumulator = AllDataAggregator.add(self.accumulator, result)

            file_path = self.data_dir / get_json_file_name(
                self.part_ind, self.batch_ind, column_data['id'])
            with open(file_path, 'w') as f:
                json.dump(column_data, f)
            self.written_files.append(file_path)
        else:
            raise ValueError('I dont like this id value')

    def finish(self) -> Optional[Tuple[List[Path], Monoid]]:
        return self.written_files, self.accumulator


class PickyShardSpec(sharded_dataset.ShardSpec):
    def __init__(self, fetch_columns: List[str]) -> None:
        self._fetch_columns = fetch_columns

    @property
    def fetch_columns(self) -> List[str]:
        return self._fetch_columns

    def get_producer(
            self,
            target_file_directory: Path,
            partition_id: int,
            batch_id: int
    ) -> PickyJsonShardProducer:
        return PickyJsonShardProducer(
            target_file_directory, partition_id, batch_id)

    @property
    def aggregation_monoid(self) -> type:
        return AllDataAggregator


@pytest.mark.asyncio
async def test_data_to_shards(
        spark_session: SparkSession,
        temp_directory: Path,
        aio_s3_client: aiobotocore.client.AioBaseClient,
        aio_s3_client_kwargs: Dict[str, Any],
        event_loop: asyncio.BaseEventLoop,
        create_bucket: Callable[[str, str], Awaitable[str]],
        aws_region: str
) -> None:
    count = 120
    ids = np.arange(count)
    int_column_values = np.random.randint(1000, size=count)
    str_column_values = list(map(lambda v: "value_" + str(v), ids))

    bucket_name = await create_bucket()

    s3_paths_pairs = []
    s3_paths_single = []
    bodies_pairs = []
    bodies_single = []

    upload_jobs = []

    # initialize s3 state
    for id in ids:
        s3_key_0 = f'data/{id}_part_0'
        s3_key_1 = f'data/{id}_part_1'
        s3_key_s = f'data/{id}_s'
        body_0 = f'{id}_0'.encode('ascii')
        body_1 = f'{id}_1'.encode('ascii')
        body_s = f'{id}_s'.encode('ascii')

        s3_paths_pairs.append(
            [f'/{bucket_name}/{s3_key_0}', f'/{bucket_name}/{s3_key_1}'])
        s3_paths_single.append(f'/{bucket_name}/{s3_key_s}')

        bodies_pairs.append([body_0, body_1])
        bodies_single.append(body_s)

        for body, key in [(body_0, s3_key_0), (body_1, s3_key_1), (body_s, s3_key_s)]:
            upload_jobs.append(
                aio_s3_client.put_object(
                    Body=body, Bucket=bucket_name, Key=key))

    await asyncio.wait(upload_jobs)

    data_pd = pd.DataFrame(
        data=dict(
            id=ids,
            value=int_column_values,
            name=str_column_values,
            links2pair=s3_paths_pairs,
            links2single=s3_paths_single
        ))

    par_level = 4
    par_config = ParallelismConfig(
        keep_partitions=False, mult=1, parallelism_level=par_level)
    data_df = spark_session.createDataFrame(data_pd)

    columns_to_fetch = ['links2pair', 'links2single']
    shard_spec = PickyShardSpec(columns_to_fetch)
    examples_per_shard = 5
    target_key_prefix = 'upload'

    job_results, agg_info = sharded_dataset.data_to_shards(
        spark_session,
        data_df,
        target_location=FileLocation(
            Path('/') / bucket_name / target_key_prefix,
            FileSource.S3),
        shard_spec=shard_spec,
        examples_per_shard=examples_per_shard,
        par_config=par_config,
        aio_boto_s3_client_kwargs=aio_s3_client_kwargs)

    # ---- check job results ---------------------------------------------------------
    assert job_results.total_examples == count
    # 2/3 of examples should fail
    assert job_results.failed_examples == (2 * (count // 3))
    # there should be as many files as successful locations
    assert len(job_results.shard_locations) == (count // 3)

    for shard_loc in job_results.shard_locations:
        assert shard_loc.file_source == FileSource.S3
        assert shard_loc.path.parent == (Path('/') / bucket_name / target_key_prefix)
        assert shard_loc.path.name.split('.')[1] == 'json'

    # ---- check uploaded data ------------------------------------------------------
    async def get_data_row(file_location: FileLocation) -> Awaitable[pd.Series]:
        parts = file_location.path.parts
        bucket = parts[1    ]
        key = '/'.join(parts[2:])
        stream = (await aio_s3_client.get_object(Bucket=bucket, Key=key))['Body']
        async with stream as s:
            data = await s.read()

        column_data = json.loads(data)
        return pd.Series(index=list(column_data),
                         data=list(column_data.values()))

    data_jobs = list(map(get_data_row, job_results.shard_locations))

    s3_data_pd = pd.DataFrame(await asyncio.gather(*data_jobs)) \
                   .sort_values(by='id') \
                   .sort_index(axis=1) \
                   .reset_index(drop=True)
    expected_data_pd = data_pd.iloc[::3] \
                              .sort_index(axis=1) \
                              .reset_index(drop=True)

    assert (expected_data_pd == s3_data_pd).all(axis=None)

    # ---- check aggregation results ---------------------------------------------
    expected_s3_data = pd.DataFrame(
        data=dict(
            id=ids,
            links2pair=bodies_pairs,
            links2single=bodies_single)) \
                         .iloc[::3] \
                         .sort_index(axis=1) \
                         .reset_index(drop=True)

    agg_s3_data = agg_info.s3_data \
        .sort_values(by='id') \
        .sort_index(axis=1) \
        .reset_index(drop=True)

    assert (expected_s3_data == agg_s3_data).all(axis=None)


def test_sharded_dataset_pb_serialization(
        temp_directory: Path
) -> None:

    dataset_info = sharded_dataset.ShardedDatasetInfo(
        splits={
            'train': sharded_dataset.ShardsSplitInfo(
                shard_locations=[
                    FileLocation(Path('/path/to/train/a.txt'), FileSource.LOCAL),
                    FileLocation(Path('/path/to/train/b.txt'), FileSource.S3)
                ],
                examples_count=100,
                not_included_count=45),
            'test': sharded_dataset.ShardsSplitInfo(
                shard_locations=[
                    FileLocation(Path('/path/to/test/a.txt'), FileSource.HDFS),
                    FileLocation(Path('/path/to/test/b.txt'), FileSource.LOCAL)
                ],
                examples_count=30,
                not_included_count=15),
        },
        split_by_column='some_label_column')

    out_path = temp_directory / 'dataset_info.pb'
    file_handler = FileHandler()
    save_as_proto_txt_file(
        dataset_info,
        FileLocation(out_path, FileSource.LOCAL),
        file_handler
    )
    read_info = read_from_proto_txt_file(
        sharded_dataset.ShardedDatasetInfo,
        FileLocation(out_path, FileSource.LOCAL),
        file_handler
    )

    assert read_info.split_by_column == dataset_info.split_by_column
    assert sorted(read_info.splits) == sorted(dataset_info.splits)
