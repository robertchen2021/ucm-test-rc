import pytest
from nauto_datasets.core.sensors import ImuStream, CombinedRecording
from nauto_zoo import MalformedSensorGZipError
from nauto_zoo.preprocess.sensor import SensorPreprocessorImu, SensorPreprocessorCombined, SensorCoachablePreprocessor
from pathlib import Path
from typing import List


def test_should_parse_sensor_imu():
    sut = SensorPreprocessorImu()
    sensor: ImuStream = sut.preprocess_sensor_files([get_test_sensor_path()])
    assert type(sensor) is ImuStream
    assert sensor.acc.x.shape[0] == 1998
    assert sensor.gyro.x.shape[0] == 1998


def test_oriented_to_raw_conversion():
    sut = SensorPreprocessorCombined()
    com_rec: CombinedRecording = sut.preprocess_sensor_files([get_test_sensor_path_raw_empty()])
    assert getattr(com_rec, 'rt_acc').stream.x.shape[0] == 1994
    assert abs(round(getattr(com_rec, 'rt_acc').stream.x[0], 3) - 4.452) < 0.001


def test_should_parse_sensor_combined():
    sut = SensorPreprocessorCombined()
    sensor: CombinedRecording = sut.preprocess_sensor_files([get_test_sensor_path()])
    assert type(sensor) is CombinedRecording
    imu = ImuStream.from_recording(sensor)
    assert type(imu) is ImuStream
    assert imu.acc.x.shape[0] == 1998
    assert imu.gyro.x.shape[0] == 1998


def test_sensor_combined_should_raise_MalformedSensorGZipError():
    sut = SensorPreprocessorCombined()
    with pytest.raises(MalformedSensorGZipError):
        sut.preprocess_sensor_files(get_test_sensor_path_bad_gzip_file())


def test_sensor_coachable_should_parse():
    sut = SensorCoachablePreprocessor()
    sample_input: List = sut.preprocess_sensor_files(get_test_sensor_path_coachable())
    assert len(sample_input) == 1  # only generate the input for 1 event at a time
    assert len(sample_input[0]) == 6  # for each event, the input is a list of 6 ndarrays
    # batch_size always = 1, since we generate input for 1 event at a time
    # default window_len = 81
    assert sample_input[0][0].shape == (1, 81, 3)  # oriented_acc input with shape (batch_size, window_len, 3)
    assert sample_input[0][1].shape == (1, 81, 3)  # oriented_gyro input with shape (batch_size, window_len, 3)
    assert sample_input[0][2].shape == (1, 81, 1)  # gps_speed input with shape (batch_size, window_len, 1)
    assert sample_input[0][3].shape == (1, 81, 9)  # dist_multilabel input with shape (batch_size, window_len, 9)
    assert sample_input[0][4].shape == (1, 81, 1)  # tailgating_distance input with shape (batch_size, window_len, 1)
    assert sample_input[0][5].shape == (1, 81, 1)  # fcw_ttc input with shape (batch_size, window_len, 1)


def get_test_sensor_path() -> str:
    return str(
        Path('./test_data/31a61ccb6646e2dc-16608f33b4e86419/sensor/4b05ca2a3d17cfd5aa03f389b82c154172903a88').resolve())


def get_test_sensor_path_raw_empty() -> str:
    return str(
        Path('./test_data/34ef8258c1bbbe02-16df5fd54d56fefb/sensor/a9f9f4cdce5663fe2f56481775a4d195657de60a').resolve())


def get_test_sensor_path_bad_gzip_file() -> List[str]:
    return [str(Path("./test_data/3c04038f22b90c2b-16ee66dd2610ee00/sensor/0a94545e22ea84924658c825f792a82769f91339")
                .resolve()),
            str(Path("./test_data/3c04038f22b90c2b-16ee66dd2610ee00/sensor/50a48fa75be5fd75116348c12540515d9a4b4fa6")
                .resolve())]


def get_test_sensor_path_coachable() -> List[str]:
    return [str(Path("./test_data/31a61ccb6646e2dc-16608f33b4e86419/sensor/4b05ca2a3d17cfd5aa03f389b82c154172903a88")
                .resolve()),
            str(Path("./test_data/31a61ccb6646e2dc-16608f33b4e86419/sensor/4bc1e7c4bba09de243054e479317b198592a0293")
                .resolve()),
            str(Path("./test_data/31a61ccb6646e2dc-16608f33b4e86419/sensor/38d3ac121d3775b2f850dd8fafe981fd469eaea3")
                .resolve()),
            str(Path("./test_data/31a61ccb6646e2dc-16608f33b4e86419/sensor/94b26a111f666d9f7e4462144fc2a44d4e19be65")
                .resolve()),
            str(Path("./test_data/31a61ccb6646e2dc-16608f33b4e86419/sensor/c0e2a1a6bf30c0795256b28d9d5020d03df32e9c")
                .resolve())]
