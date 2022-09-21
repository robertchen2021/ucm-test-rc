import gzip
import json
import logging
from pathlib import Path
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Tuple,
                    Union)

from pyspark.sql import SparkSession

from nauto_datasets.core.dataset import DatasetInstance, DatasetRepresentation
from nauto_datasets.core.sensors import CombinedRecording, Recording
from nauto_datasets.core.serialization import FileLocation
from nauto_datasets.core.spark import ParallelismConfig
from nauto_datasets.reprs import sharded_dataset
from nauto_datasets.serialization.jsons import sensors as sensors_ser
from nauto_datasets.utils import protobuf
from nauto_datasets.utils.category import DummyMonoid, Monoid
from sensor import sensor_pb2

"""
Function taking arguments in the following order:
- json value with the current representation of the results,
  including sensorfile data
- values of the columns requested by the `JsonValueExtractor`
"""
JsonValueExtractorFn = Callable[..., None]


class JsonValueExtractor(NamedTuple):
    """
    Attributes:
        arg_columns: names of the columns with values required
            by the `extractor_fn`. The extractor should be later
            fed with json and values of these columns the same order
            as they are provided
        extractor_fn: a function adding fields to a json reprenting a single
            dataset example
    """
    arg_columns: List[str]
    extractor_fn: JsonValueExtractorFn


_TARGET_FILE_COL_NAME = '_target_file_name'
JSON_REPR_COL_NAME = 'target_json'


class JsonConfig(NamedTuple):
    file_name_fn: Callable[[Dict[str, Any]], str]
    sensor_paths_column: str
    compress: bool
    value_extractors: Optional[List[JsonValueExtractor]] = None


class JsonProducer(sharded_dataset.ShardProducer):

    def __init__(
            self,
            target_dir: Path,
            json_config: JsonConfig
    ) -> None:
        self._target_dir = target_dir
        self._config = json_config
        self._file_list = []

    def write(self,
              column_data: Dict[str, Any],
              s3_data: Dict[str, Union[bytes, List[bytes]]]) -> None:
        file_name = self._config.file_name_fn(column_data)

        sensor_data = s3_data[self._config.sensor_paths_column]
        com_rec = CombinedRecording.from_recordings([
            Recording.from_pb(
                protobuf.parse_message_from_gzipped_bytes(
                    sensor_pb2.Recording, msg_bytes))
            for msg_bytes in sensor_data
        ])
        sensor_json = sensors_ser.combined_recording_to_json(com_rec)

        if self._config.value_extractors is not None:
            for v_e in self._config.value_extractors:
                args = [column_data[col_name] for col_name in v_e.arg_columns]
                v_e.extractor_fn(sensor_json, *args)

        if self._config.compress:
            open_f = gzip.open
            file_name += '.gz'
        else:
            open_f = open
        file_path = self._target_dir / file_name
        with open_f(str(file_path), 'w') as f:
            json_bytes = json.dumps(sensor_json).encode()
            f.write(json_bytes)
        self._file_list.append(file_path)

    def finish(self) -> Optional[Tuple[List[Path], Monoid]]:
        return self._file_list, DummyMonoid()


class JsonShardSpec(sharded_dataset.ShardSpec):

    def __init__(self, json_config: JsonConfig) -> None:
        self._config = json_config

    @property
    def fetch_columns(self) -> List[str]:
        return [self._config.sensor_paths_column]

    def get_producer(
            self,
            target_file_directory: Path,
            partition_id: int,
            batch_id: int
    ) -> JsonProducer:
        return JsonProducer(target_file_directory, self._config)

    @property
    def aggregation_monoid(self) -> type:
        return DummyMonoid


class SensorJsonsDataset(DatasetRepresentation):
    """Representation of the dataset based on sensor recordigns.

    Each row of the dataset instance will correspond to an individual
    json file. The jsons produced are simply serialized `CombinedRecordings`
    transformed by `JsonValueExtractors`.
    """

    def __init__(self,
                 compress: bool = True) -> None:
        """Create `SensorJsonsDataset`

        Args:
            compress: whether each resulting json should be gzip compressed
        """
        self._compress = compress

    def create(
            self,
            sess: SparkSession,
            di: DatasetInstance,
            dataset_directory: FileLocation,
            sensor_paths_column: str,
            file_name_fn: Callable[[Dict[str, Any]], str],
            value_extractors: Optional[List[JsonValueExtractor]] = None,
            partitioning_column: Optional[str] = None,
            processing_batch_size=64,
            par_config: ParallelismConfig = ParallelismConfig(
                keep_partitions=True)
    ) -> sharded_dataset.ShardedDatasetInfo:
        """Creates and saves the json representation of the sensor recording
        based dataset.

        Args:
            sess: spark session
            di: dataset instance to transform
            dataset_directory: the location where the splits should be saved
            sensor_paths_column: the name of the column with links
                to sensor media files on S3
            file_name_fn: function returning a json filename given all the columns
                from the dataset instance
            value_extractors: an optional list of json value extractors
                adding additional fields to produced jsons, e.g. labels
                or meta info
            partitioning_column: if provided then each split will be additionaly
                broken into two parts represented as subdirectories of names:
                "<partitioning_column>=True" and "<partitioning_column>=False"
            processing_batch_size: the maximum number of json produced concurrently
            par_config: parallelism config determining the number of partitions
                to run the data saving jobs.
        Returns:
            a dictionary mapping the split name to the number of files written
            under this split
        """
        logging.info('Producing sensor jsons dataset as a sharded dataset')

        json_config = JsonConfig(
            file_name_fn=file_name_fn,
            sensor_paths_column=sensor_paths_column,
            compress=self._compress,
            value_extractors=value_extractors)

        shards_info, _ = sharded_dataset.create_sharded_dataset(
            sess=sess,
            di=di,
            target_location=dataset_directory,
            shard_spec=JsonShardSpec(json_config),
            split_by_column=partitioning_column,
            examples_per_shard=processing_batch_size,
            par_config=par_config
        )
        return shards_info
