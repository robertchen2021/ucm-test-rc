import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as dtype

from nauto_datasets.core.dataset import (DatasetDescription, DatasetInstance,
                                         DataSource)
from nauto_datasets.core.schema import RecordSchema
from nauto_datasets.core.serialization import FileLocation, FileSource
from nauto_datasets.core.splits import RandomSplitByColumn, SplitConfig
from nauto_datasets.serialization.parquet import ParquetData, ParquetSerializer
from nauto_datasets.utils.tests import SparkTestCase

TOTAL_COUNT = 300


class DataSourceEx(DataSource):

    def produce(self,
                sess: SparkSession,
                since: Optional[datetime] = None,
                until: Optional[datetime] = None) -> DataFrame:
        ids = np.arange(TOTAL_COUNT, dtype=np.int64)
        features = np.random.rand(TOTAL_COUNT)

        label = np.repeat(False, TOTAL_COUNT)
        label[np.random.choice(TOTAL_COUNT, size=100, replace=False)] = True

        pdf = pd.DataFrame(
            dict(id=ids,
                 feature=features,
                 label=label))
        pdf.set_index('id')

        return sess.createDataFrame(pdf)

    def validate(self,
                 sess: SparkSession,
                 dataset: DataFrame,
                 valid_col_name: str = 'valid') -> DataFrame:
        return dataset.withColumn(valid_col_name, F.lit(True))

    def record_schema(self) -> RecordSchema:
        return RecordSchema(
            entities={
                'ids': [dtype.StructField('id', dtype.LongType(), False)],
                'features': [dtype.StructField('feature', dtype.DoubleType(), False)],
                'labels': [dtype.StructField('label', dtype.BooleanType(), False)],
            })


class TestDatasetInstance(SparkTestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_instance_creation(self):
        dataset_desc = DatasetDescription(
            data_source=DataSourceEx(),
            split_strategy=RandomSplitByColumn(
                split_confs=[SplitConfig('train', 2, True),
                             SplitConfig('valid', 1, True),
                             SplitConfig('test', 1, True)],
                column='label'))

        path = Path(self.tmpdir) / 'dataset'
        location = FileLocation(path=path, file_source=FileSource.LOCAL)
        name = 'TestDataset'
        since = None
        until = datetime.now()
        data_serializer = ParquetSerializer(self.spark)
        partition_columns = ['label']

        di = DatasetInstance.create(
            sess=self.spark,
            name=name,
            location=location,
            dataset_desc=dataset_desc,
            since=since,
            until=until,
            data_serializer=data_serializer,
            partition_columns=partition_columns)

        self.assertEqual(di.name, name)
        self.assertEqual(di.data_until, until)
        self.assertGreater(di.creation_time, until)
        self.assertLess(di.creation_time, datetime.now())
        self.assertSetEqual(
            set(di.splits.keys()),
            set(['train', 'valid', 'test'])
        )

        splits = di.load(self.spark)

        self.assertEqual(
            splits['train'].count() + splits['test'].count() + splits['valid'].count(),
            TOTAL_COUNT)

    def test_instance_update(self):
        dataset_desc = DatasetDescription(
            data_source=DataSourceEx(),
            split_strategy=RandomSplitByColumn(
                split_confs=[SplitConfig('train', 2, True),
                             SplitConfig('valid', 1, True),
                             SplitConfig('test', 1, True)],
                column='label'))

        path = Path(self.tmpdir) / 'old_dataset'
        location = FileLocation(path=path, file_source=FileSource.LOCAL)
        name = 'OldDataset'
        since = datetime.now() - timedelta(weeks=10)
        until = datetime.now() - timedelta(weeks=3)
        data_serializer = ParquetSerializer(self.spark)
        partition_columns = ['label']

        old_di = DatasetInstance.create(
            sess=self.spark,
            name=name,
            location=location,
            dataset_desc=dataset_desc,
            since=since,
            until=until,
            data_serializer=data_serializer,
            partition_columns=partition_columns)

        new_path = Path(self.tmpdir) / 'new_dataset'
        new_location = FileLocation(path=new_path, file_source=FileSource.LOCAL)
        new_name = 'NewDataset'
        new_until = datetime.now() - timedelta(weeks=1)

        new_di = DatasetInstance.update(
            sess=self.spark,
            old_di=old_di,
            name=new_name,
            location=new_location,
            dataset_desc=dataset_desc,
            until=new_until,
            data_serializer=data_serializer,
            partition_columns=partition_columns)

        self.assertEqual(new_di.name, new_name)
        self.assertEqual(new_di.data_until, new_until)
        self.assertGreater(new_di.creation_time, new_until)
        self.assertLess(new_di.creation_time, datetime.now())
        self.assertSetEqual(
            set(new_di.splits.keys()),
            set(['train', 'valid', 'test'])
        )

        splits = new_di.load(self.spark)

        self.assertEqual(
            splits['train'].count() + splits['test'].count() + splits['valid'].count(),
            2 * TOTAL_COUNT)

    def test_pb_serialization(self):

        di = DatasetInstance(
            name='TestData',
            creation_time=datetime.now(),
            data_since=datetime.now() - timedelta(weeks=4),
            data_until=datetime.now() - timedelta(weeks=1),
            schema=RecordSchema(
                entities={
                    'ids': [dtype.StructField('id', dtype.IntegerType(), False)],
                    'features': [dtype.StructField('feature', dtype.DoubleType(), False)],
                    'labels': [dtype.StructField('label', dtype.BooleanType(), False)],
                }),
            splits={
                'train': ParquetData(
                    location=FileLocation(
                        path=Path('/data/training'),
                        file_source=FileSource.S3
                    ),
                    compression='gzip'
                ),
                'valid': ParquetData(
                    location=FileLocation(
                        path=Path('/data/validation'),
                        file_source=FileSource.LOCAL
                    ),
                    compression='snappy'
                ),
                'test': ParquetData(
                    location=FileLocation(
                        path=Path('/data/testing'),
                        file_source=FileSource.HDFS
                    ),
                    compression='gzip',
                    partitioning_columns=['labels']
                )
            })

        di_pb = di.to_pb()
        des_di = DatasetInstance.from_pb(di_pb)

        self.assertEqual(di.name, des_di.name)
        self.assertEqual(di.creation_time, des_di.creation_time)
        self.assertEqual(di.data_since, des_di.data_since)
        self.assertEqual(di.data_until, des_di.data_until)
        self.assertDictEqual(di.schema.entities, des_di.schema.entities)
        # self.assertDictEqual(di.schema.splits, des_di.schema.splits)
        self.assertSetEqual(
            set(di.splits.keys()), set(des_di.splits.keys()))

        for split_name in di.splits:
            self.assertEqual(
                di.splits[split_name].location,
                des_di.splits[split_name].location)
            self.assertEqual(
                di.splits[split_name].compression,
                des_di.splits[split_name].compression)
            self.assertEqual(
                di.splits[split_name].partitioning_columns,
                des_di.splits[split_name].partitioning_columns)
