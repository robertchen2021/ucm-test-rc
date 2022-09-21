# Installation

The library comes with some extra dependencies, which are required only for a subset of usecases.
Due to the weight of the corresponding libraries, they were not included in the regular package.

## For development
```sh
pip install -e .[tf-cpu,spark,dev]
```
or
```sh
pip install -e .[tf-gpu,spark,dev]
```

## Packaging
In order to build a wheel package run:
```sh
python setup.py bdist_wheel
```

Installing the wheel:
```sh
pip install <path-to-wheel>[tf-cpu,spark]
```
or
```sh
pip install <path-to-wheel>[tf-gpu,spark]
```

## Install as dependency
```sh
pip3 install --index-url=https://${ARTIFACTORY_USER}:${ARTIFACTORY_PASSWORD}@nauto.jfrog.io/nauto/api/pypi/drt-virtual/simple/ nauto-datasets
```

## Other dependencies

`nauto-datasets` installed with pip statement delivers all python dependencies declared with `setup.py`.
> Pay attention NOT to use `ffmpeg` package in addition to `nauto-datasets` as it clashes with `ffmpeg-python`

Apart from this, the native `ffmpeg` system dependency is required to run functionality, which deals with video
content. [FFmpeg](#http://ffmpeg.org/download.html) installation is system specific. Version `4.2.1` was used 
successfully so far.

# Generating datasets

Datasets generation is based on events input query. It's up to the user how many fields query returns, however there
are certain requirements on minimal number of inputs to return.
It's required for query to return `message_params` (drt) or `params` (ep). It's also required to return basic metadata
(see examples below). Vehicle information is not required, but highly recommended.

## Qubole notebook

In order to run in `qubole` notebook, the following variables have to be added to `spark` interpreter:
```
spark.pyspark.driver.python=/usr/lib/anaconda2/envs/py36/bin/python
spark.pyspark.python=/usr/lib/anaconda2/envs/py36/bin/python
```

In notebook write the following:

```python
from nauto_datasets.drt.reprs.ds_data import DRTEvents, PackagedEvents
from nauto_datasets.drt.reprs.ds_config import RequestConfig
from nauto_datasets.drt.reprs.ds_format import DSFormatter

# provide events query first
# it's required to include sensor and vehicle data in the output
query = """
        SELECT s.*,
                  v.vin AS vin,
                  regexp_replace(regexp_replace(trim(lower(v.make)), ' ', '_'), '_&_', '_and_') AS make,
                  regexp_replace(regexp_replace(trim(lower(v.model)), ' ', '_'), '_&_', '_and_') AS model,
                  v.year
             FROM drt.judgements AS j
        LEFT JOIN drt.events AS e
               ON j.event_id = e.id
        LEFT JOIN device.severe_g AS s
               ON LOWER(HEX(e.device_id)) || '-' || LOWER(HEX(e.message_id)) = s.device_id || '-' || s.message_id
        LEFT JOIN dimension.vehicle AS v
               ON s.vehicle_id = v.vehicle_id
            WHERE j.created_at >= '2020-01-01'
              AND j.task_type = 'crashnet'
              AND (get_json_object(j.info, '$.near-collision-subquestion') like '%["pedestrian"]%'
                   OR get_json_object(j.info, '$.risky-manuever-subquestion') like '%["pedestrian"]%'
                   OR get_json_object(j.info, '$.what-did-hit') like '%["pedestrian"]%')
              AND s.message_type IN ('crashnet', 'severe-g-event')
"""
# wrap query around Events object
events = DRTEvents(query=query, spark=spark)
# alternatively packaged_events based queries can be used, which are usually faster
packaged_events_query = """
   SELECT pe.id, pe.type, pe.device_id, pe.message_id, pe.message_ts, pe.fleet_id, pe.s2_cell_id, 
            pe.accepted_at, pe.received_at, pe.created_at, pe.params,
                 v.vin AS vin,
                  regexp_replace(regexp_replace(trim(lower(v.make)), ' ', '_'), '_&_', '_and_') AS make,
                  regexp_replace(regexp_replace(trim(lower(v.model)), ' ', '_'), '_&_', '_and_') AS model,
                  v.year
             FROM event_packager.packaged_events AS pe
        LEFT JOIN dimension.vehicle AS v
               ON get_json_object(pe.params, '$.vehicle_id') = v.vehicle_id
            WHERE pe.message_day >= '2020-01-01' AND pe.message_day <= '2020-01-20'
              AND pe.type IN ('crashnet', 'severe-g-event')
              AND (
                      (pe.near_collision_subquestion.confidence=1 AND pe.near_collision_subquestion.value like '%["pedestrian"]%')
                  OR  (pe.risky_maneuver_subquestion.confidence=1 AND pe.risky_maneuver_subquestion.value like '%["pedestrian"]%')
                  OR  (pe.what_did_hit.confidence=1 AND pe.what_did_hit.value like '%["pedestrian"]%')
                  )
"""
events = PackagedEvents(query=packaged_events_query, spark=spark)

# once called events.get() you can retrieve events from database - this is time consuming operation,
# but events are cached inside Events object instance

# prepare request config instance, all required attributes are listed below
config = RequestConfig(prepare_new_request=True,
                       request_description='This is test dataset generation',
                       requested_by='john_smith@nauto.com',
                       prepared_by='foo_bar@nauto.com',
                       dataset_name='test_dataset_with_mock_data')

# setup formatter to indicate where the data is flowing
formatter = DSFormatter(project_name='20200122-project-you-are-working-on',
                        project_bucket='nauto-prod-ai',
                        project_key='work/my_user_name', 
                        spark=spark, in_qubole=True)

# launch
formatter.generate_data_set(config, events)
``` 

### Custom input

In case input requires some further transformation before generating data set, or the data is already provided from
 some upstream computation, it's possible to trigger dataset generation by providing this source explicitly.

```python
from nauto_datasets.drt.reprs.event_set import EventSource, EventSet
from nauto_datasets.drt.reprs.ds_config import RequestConfig
import pandas as pd
from typing import Dict

class Source(EventSource):

    def __init__(self):
        # this should be initialized with metadata information that's important for this processing
        # it will be additionally filled with additional processing data during dataset generation
        self._events_metadata = {}

    @property
    def events(self) -> pd.DataFrame:
        return 'pandas data from containing events'

    @property
    def videos(self) -> pd.DataFrame:
        return f'pandas data from containing videos ids as array of strings, in column pointed by self.video_column '
        'having message_id and device_id column that will be used for matching against self.events'

    @property
    def sensors(self) -> pd.DataFrame:
        return f'pandas data from containing sensors ids as array of strings, in column pointed by self.sensor_column '
        'having message_id and device_id column that will be used for matching against self.events'

    @property
    def request_config(self) -> RequestConfig:
        return RequestConfig(prepare_new_request=True,
                       request_description='This is test dataset generation from custom input',
                       requested_by='john_smith@nauto.com',
                       prepared_by='foo_bar@nauto.com',
                       dataset_name='test_dataset_with_mock_data')

    @property
    def events_metadata(self) -> Dict:
        return self._events_metadata

source = Source()
event_set = EventSet(source, project_bucket='nauto-prod-ai', project_key='work/my_user_name',
            project_name='20200122-project-you-are-working-on', spark=spark)
# this step will generate media files on S3
processed_events, processed_events_metadata = event_set.create()
# this step will save meta information about processed events and request
event_set.upload()
```

## CLI

Having library installed run the following:

```shell script
nauto-datasets generate --qubole-token $QUBOLE_AUTH_TOKEN --cluster $CLUSTER_LABEL \
               ds --name $PROJECT_NAME --key $PROJECT_KEY --source drt \
                  --query <path_to_file_with_query> --config <path_to_file_with_config>
```

Use https://us.qubole.com/v2/control-panel#manage-accounts to check your `qubole` token.
Cluster requires `nauto_datasets>=0.6` and python 3.6 installed.
The output path is generated to `s3://nauto-prod-ai/$PROJCT_KEY/data/processed/$PROJECT_NAME`.
Given above example, dataset is created in location `s3://nauto-prod-ai/work/my_user_name/data/processed/20200122-project-you-are-working-on`

## Configuration

Configuration is either specified via API with `RequestConfig` class, or read from yaml file. The following fields
are supported:

```yaml
prepare_new_request: true             # if true, whole dataset generation is done from scratch, otherwise metadata is read from S3 and only media files are downloaded
request_description: 'what is inside' # tells what kind of information is stored in datasets in human readable form
requested_by: 'login.of.requestor'    # who has requsted the dataset
prepared_by: 'login.of.creator'       # who has prepared the dataset
dataset_name: 'name_of_the_dataset'   # logical name of this dataset    
rights: 'rights clause'               # [optional] allows to override default nauto clause
join_videos: 'both'                   # [optional] how to join videos, choices are:
                                      #  - 'both'  to output joined and separate (default)
                                      #  - 'true'  to join interior/exterior videos
                                      #  - 'false' to output interior/exterior separately
join_videos_type: 'horizontal-ext-int'# [optional] how to align videos for joining, choices are: horizontal-ext-int (default), horizontal-int-exet, vertical-ext-int, vertical-int-ext
extract_audio: false                  # [optional] if audio should be extracted or not
create_sensor_json: true              # [optional] if sensor json file should be created
plot_sensor_data: true                # [optional] if sensor information should be put into plot
dry_run: false                        # [optional] if true, results are not persisted to S3
metadata_cols:                        # [optional] list of columns to expect in events' input
  - 'fleet_id'
  - 'device_id'
  - 'message_id'
  - 'message_type'
  - 'message_ts'
  - 'vehicle_id'
  - 'vin'
  - 'make'
  - 'model'
  - 'year'
  - 'vehicle_type'
  - 'utc_basetime'
handle_failure_cases:                 # [optional] how to handle particular failure cases
  - 'failed_video_sensor_time'        # (default) is_sensor_time=0` in message_params, meaning firmware callback is not used to assign sensor_time (output may not be well synced)   
  - 'failed_video_duration_valid'     # (default) sensor_start time is incorrect from video cutoff bug, so reconstruct start_time using sensor_end - video_trans_duration (output may not be well synced)
  - 'failed_media_downloaded'         # some videos or sensor files did not download, attempt to stitch with gaps
```

# Managing sensors API

## Updating sensors proto

`sensor/sensor.proto` file is used to generate `sensor/sensor_pb2.py`, which includes mapping of sensor data structures.
The generation is done during packaging of the library. `nauto_datasets/core/sensors.py` makes use of the proto generated
structures to expose them as `nauto_datasets.core.SensorStream` based classes. Update of `sensors.py` has to be done
manually when content of `sensor/sensor.proto` is updated, to reflect the new sensor information.

The `sensor/sensor.proto` is a symlink to `nauto-ai/schema/protos/protobuf/sensors/sensor.proto`. Make sure to update the submodule (please read the parent directory, `nauto-ai`'s ReadMe) with the appropriate module version by running
```shell
make genprotos
```
from the root directory.

```shell
make local_install
```
This will regenerate `sensor/sensor_pb2.py` making it available to `nauto_datasets/core/sensors.py` for adaptation.

## Creating new sensor representation
Before adding a new sensor representation, please check if the stream is present in the sensor protos (`../schema/protos`), has the sensor stream inside the protobuf. If so, the changes refer to file `nauto_datasets/core/sensors.py`.

Firstly, new class should be created representing new stream:
```python
class NewlyCreatedStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    any_array_typed_field_from_protobuf: NDArray[np.float]
    another_array_typed_field_from_protobuf: NDArray[np.unit64]
    ...

    @staticmethod
    def from_pb(nc_pb: sensor_pb2.NewlyCreatedStream) -> 'NewlyCreatedStream':
        return _get_named_tuple_from_pb(NewlyCreatedStream, nc_pb)
```
`sensor_ns` and `system_ms` are obligatory for `SensorStream`.

Secondly, `Recording` class should be modified with new fields (in case it's included in protobuf):
```python
class Recording(metaclass=NamedTupleMetaEx):
...
    newly_created: NewlyCreatedStream
...
    @staticmethod
    def from_pb(r_pb: sensor_pb2.Recording) -> 'Recording':
        ...
        newly_created=_maybe_empty_stream(NewlyCreatedStream, r_pb.newly_created)
        ...
```

Then create combined stream type:
```python
ComNewlyCreatedStream = streams.create_combined_stream_type(
    'ComNewlyCreatedStream',
    NewlyCreatedStream,
    [CombinedUtcTimeConvertible])
```

and add it to combined recording:
```python
class CombinedRecording(metaclass=NamedTupleMetaEx):
...
    newly_created: ComNewlyCreatedStream
...
    @staticmethod
    def from_recordings(recordings: List[Recording]) -> 'CombinedRecording':
...
              newly_created=ComNewlyCreatedStream.from_substreams(
            [rec.newly_created for rec in recordings]),
...
```
# Releasing a new version

`nauto_datasets/__init__.py` includes `__version__` variable that has to be incremented.
Merging changes to master releases new version of the library.

> Remember to upgrade the version number every time the code changes.

# Common issues

## MacOS

* on running `Spark` related tests, when encountered:
    ```
          [...]
        py4j.protocol.Py4JJavaError: An error occurred while calling z:org.apache.spark.api.python.PythonRDD.collectAndServe.
        E                   : org.apache.spark.SparkException: Job aborted due to stage failure: Task 2 in stage 1.0 failed 1 times, most recent failure: Lost task 2.0 in stage 1.0 (TID 6, localhost, executor driver): org.apache.spark.SparkException: Python worker exited unexpectedly (crashed)
          [...]
    ```
    add environment variable: `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`

