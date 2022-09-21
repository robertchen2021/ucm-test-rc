from pathlib import Path
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union)

import tensorflow as tf

from nauto_datasets.reprs.sharded_dataset import ShardProducer, ShardSpec
from nauto_datasets.serialization.tfrecords import encoding as tf_encoding
from nauto_datasets.utils.category import Monoid, DummyMonoid
from nauto_datasets.utils.tuples import NamedTupleMetaEx

TfFeaturesProducer = Callable[
    # Takes
    [
        Dict[str, Any],                       # column data
        Dict[str, Union[bytes, List[bytes]]]  # s3 downloaded data
    ],
    # Returns
    Tuple[
        tf_encoding.TfFeatures,  # features to write
        Monoid                   # info to aggregate
    ]
]
"""Function:
    (column_data, s3_data) -> (tf_features, aggregation_result)
"""


class TfFeaturesConfig(metaclass=NamedTupleMetaEx):
    """
    Attributes:
        get_features_producer: a factory function used to create `TfFeaturesProducer`
            This is necessary to keep it as a factory method rather than already
            created producer, because the producer will be potentially created in
            a separate node in a different process. It might contain data, which is not
            picklable, e.g. connections
        tf_record_options: options passed to TfRecordWriter
        use_sequence_examples: whether tfrecords should contain sequence examples
        aggregator_t: a `Monoid` type used to aggregate the results
            globally accross the entire data split
        columns_to_fetch: columns with links to s3 data which should be downloaded
    """
    get_features_producer: Callable[[], TfFeaturesProducer]
    tf_record_options: tf.io.TFRecordOptions
    use_sequence_examples: bool = False
    aggregator_t: type = DummyMonoid
    columns_to_fetch: Optional[List[str]] = None


class TfRecordProducer(ShardProducer):

    def __init__(
            self,
            record_file_path: Path,
            tf_features_config: TfFeaturesConfig,
    ) -> None:
        tf.compat.v1.enable_eager_execution()
        self._record_file_path = record_file_path
        self._config = tf_features_config

        self._producer = self._config.get_features_producer()

        self._writer = tf.io.TFRecordWriter(
            str(record_file_path),
            options=self._config.tf_record_options)
        self._agg = self._config.aggregator_t.zero()

    def write(self,
              column_data: Dict[str, Any],
              s3_data: Dict[str, Union[bytes, List[bytes]]]) -> bool:
        """Returns true if data was accepted and written to a shard file"""
        tf_features, agg_result = self._producer(column_data, s3_data)

        if self._config.use_sequence_examples:
            self._writer.write(
                tf_features.to_sequence_example().SerializeToString())
        else:
            self._writer.write(
                tf_features.to_example().SerializeToString())

        self._agg = self._config.aggregator_t.add(self._agg, agg_result)

    def finish(self) -> Optional[Tuple[List[Path], Monoid]]:
        """Finishes writing and returns paths to created files
        along with aggregation results.
        If result is None, then no data file has been created
        """
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            return [self._record_file_path], self._agg
        else:
            raise RuntimeError('TfRecord producer has already finished')


class TfRecordsSpec(ShardSpec):

    def __init__(
            self,
            tf_features_config: TfFeaturesConfig,
    ) -> None:
        """Creates `TfRecordsSpec` used when creating sharded datasets of
        tensorflow records

        Args:
            tf_features_config: specifying what and how to serialize the features
        """
        self._tf_features_config = tf_features_config

    @property
    def fetch_columns(self) -> List[str]:
        return self._tf_features_config.columns_to_fetch

    @property
    def aggregation_monoid(self) -> type:
        return self._tf_features_config.aggregator_t

    def get_producer(
            self,
            target_file_directory: Path,
            partition_id: int,
            batch_id: int
    ) -> TfRecordProducer:
        target_file_path = target_file_directory / f'part_{partition_id}_{batch_id}.tfrecord'
        return TfRecordProducer(
            target_file_path,
            tf_features_config=self._tf_features_config)
