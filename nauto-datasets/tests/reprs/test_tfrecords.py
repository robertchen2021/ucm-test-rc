import shutil
import tempfile
from pathlib import Path
from typing import NamedTuple, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from pyspark.sql import SparkSession

from nauto_datasets.core.serialization import FileLocation, FileSource
from nauto_datasets.core.spark import ParallelismConfig
from nauto_datasets.reprs import tfrecords as tf_repr, sharded_dataset
from nauto_datasets.serialization.tfrecords import encoding as tf_encoding
from nauto_datasets.serialization.tfrecords import decoding as tf_decoding
from nauto_datasets.utils.numpy import NDArray
from nauto_datasets.utils.category import DummyMonoid


class RecordData(NamedTuple):
    val_int: np.int64
    val_str: np.str
    val_arr: NDArray[np.float]


def get_features_config() -> tf_repr.TfFeaturesConfig:
    def produce_tf_features(
            column_data: Dict[str, Any],
            s3_data: Dict[str, Any]
    ) -> tf_encoding.TfFeatures:
        values_two = column_data['values_two']
        values_one = column_data['values_one']
        larger_arr = column_data['larger_arr']
        names = column_data['names']
        rd = RecordData(
            val_int=values_two + values_one,
            val_str=names,
            val_arr=np.array(larger_arr, dtype=np.float))
        return tf_encoding.structure_to_features(rd, use_tensor_protos=False), DummyMonoid()

    return tf_repr.TfFeaturesConfig(
        get_features_producer=lambda: produce_tf_features,
        tf_record_options=tf.io.TFRecordOptions(compression_type='GZIP'),
        use_sequence_examples=False,
        columns_to_fetch=[]
    )


def get_test_data(count: int) -> pd.DataFrame:

    data = dict(
        values_one=np.arange(count),
        values_two=np.random.randint(0, 100, count),
        names=np.array([hex(i) for i in range(count)]),
        larger_arr=pd.Series(
            [list(map(float, np.random.randn(3)))
             for i in range(count)]),
        useless=list(map(float, np.random.randn(count))))

    return pd.DataFrame(data)


class TestTfRecordsRepresentation(tf.test.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_data_to_tf_records(self):

        spark = SparkSession.builder.getOrCreate()
        total_count = 100
        examples_per_record = 6
        features_config = get_features_config()
        par_count = 4
        par_config = ParallelismConfig(False, 1, par_count)
        target_path = Path(self.tmp_dir) / 'tf_records'
        target_path.mkdir(parents=True)
        target_location = FileLocation(
            path=target_path,
            file_source=FileSource.LOCAL)

        pdf = get_test_data(total_count)
        data_df = spark.createDataFrame(pdf)

        tf_shard_spec = tf_repr.TfRecordsSpec(features_config)
        records_info, _ = sharded_dataset.data_to_shards(
            sess=spark,
            data_df=data_df,
            target_location=target_location,
            shard_spec=tf_shard_spec,
            examples_per_shard=examples_per_record,
            par_config=par_config,
            aio_boto_s3_client_kwargs=dict(
                aws_secret_access_key='xxx',
                aws_access_key_id='xxx'
            ))

        self.assertEqual(records_info.total_examples, len(pdf))
        self.assertEqual(records_info.failed_examples, 0)
        self.assertGreaterEqual(len(records_info.shard_locations), par_count)

        for loc in records_info.shard_locations:
            self.assertTrue(loc.path.exists())
            self.assertEqual(loc.path.parent, target_path)

        parsers = tf_decoding.nested_type_to_feature_parsers(
            RecordData,
            parse_tensor_protos=False,
            ignore_sequence_features=True)

        dataset = tf.data.TFRecordDataset(
            [str(loc.path) for loc in records_info.shard_locations],
            compression_type='GZIP'
        ).map(parsers.parse_example)

        count = 0
        for next_element in dataset:
            try:
                self.assertSetEqual(
                    set(next_element.keys()),
                    set(RecordData._fields))
                self.assertEqual(next_element['val_arr'].shape, (3,))
                count += 1
            except tf.errors.OutOfRangeError:
                break
        self.assertEqual(count, total_count)
