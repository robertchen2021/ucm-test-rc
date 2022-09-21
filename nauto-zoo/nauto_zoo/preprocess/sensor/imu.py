from .interfaces import AbstractSensorPreprocessor
from .combined import SensorPreprocessorCombined
from nauto_datasets.core.sensors import ImuStream, XYZStream, CombinedRecording
from typing import List, NamedTuple, Dict
import numpy as np

class OrientedImuStream(NamedTuple):
    acc: XYZStream
    gyro: XYZStream

    @staticmethod
    def from_recording(recording: CombinedRecording) -> "OrientedImuStream":
        if recording.oriented_acc.stream._is_empty() or recording.oriented_gyro.stream._is_empty():
            raise ValueError("Streams are empty!")

        gyro = recording.oriented_gyro.stream
        version = recording.metadatas[0].version
        if (
                version is not None
                and len(version) > 0
                and float(".".join(version.split(".")[:2])) < 2.4
        ):
            # there is supposedly a bug in the old versions of sensor data.
            gyro = gyro._replace(x=gyro.x * 4.0, y=gyro.y * 4.0, z=gyro.z * 4.0)

        return OrientedImuStream(acc=recording.oriented_acc.stream, gyro=gyro)

class RTImuStream(NamedTuple):
    acc: XYZStream
    gyro: XYZStream

    @staticmethod
    def from_recording(recording: CombinedRecording) -> "RTImuStream":
        if recording.rt_acc.stream._is_empty() or recording.rt_gyro.stream._is_empty():
            raise ValueError("raw streams are empty!")
        else:
            return RTImuStream(acc=recording.rt_acc.stream, gyro=recording.rt_gyro.stream)

class RTOrientedImuStream(NamedTuple):
    acc: XYZStream
    gyro: XYZStream

    @staticmethod
    def from_recording(recording: CombinedRecording) -> "RTOrientedImuStream":
        if recording.rt_oriented_acc.stream._is_empty() or recording.rt_oriented_gyro.stream._is_empty():
            raise ValueError("Oriented streams are empty!")
        else:
            return RTOrientedImuStream(acc=recording.rt_oriented_acc.stream, gyro=recording.rt_oriented_gyro.stream)

class SensorPreprocessorImu(AbstractSensorPreprocessor):
    def __init__(
            self,
            max_length=None,
            cut_beginning=False,
            clip_acc_x=None,
            clip_gyro_x=None,
            clip_acc_y=None,
            clip_gyro_y=None,
            clip_acc_z=None,
            clip_gyro_z=None,
    ):
        self._preprocessor_combined = SensorPreprocessorCombined()
        self.max_length = max_length
        self.cut_beginning = cut_beginning
        self.clip_acc_x = clip_acc_x
        self.clip_gyro_x = clip_gyro_x
        self.clip_acc_y = clip_acc_y
        self.clip_gyro_y = clip_gyro_y
        self.clip_acc_z = clip_acc_z
        self.clip_gyro_z = clip_gyro_z

    def sort_xyz_stream(self, xyzstream, sort_idx):
        xyzstream.sensor_ns[:] = xyzstream.sensor_ns[sort_idx]
        xyzstream.x[:] = xyzstream.x[sort_idx]
        xyzstream.y[:] = xyzstream.y[sort_idx]
        xyzstream.z[:] = xyzstream.z[sort_idx]

    def preprocess_sensor_files(self, sensor_files: List[str], metadata: Dict = None, oriented_streams: bool = False) -> ImuStream:
        com_rec = self._preprocessor_combined.preprocess_sensor_files(sensor_files)
        if oriented_streams:
            imu_stream = RTOrientedImuStream.from_recording(com_rec)
        else:
            imu_stream = RTImuStream.from_recording(com_rec)

        self.sort_xyz_stream(imu_stream.acc, np.argsort(imu_stream.acc.sensor_ns))
        self.sort_xyz_stream(imu_stream.gyro, np.argsort(imu_stream.gyro.sensor_ns))

        acc = imu_stream.acc
        gyro = imu_stream.gyro
        if self.max_length is not None:
            acc = acc.cut(max_length=self.max_length, cut_beginning=self.cut_beginning),
            gyro = gyro.cut(max_length=self.max_length, cut_beginning=self.cut_beginning),
        if self.clip_acc_x is not None:
            acc = acc.clip_x(max_value=self.clip_acc_x)
        if self.clip_gyro_x is not None:
            gyro = gyro.clip_x(max_value=self.clip_gyro_x)
        if self.clip_acc_y is not None:
            acc = acc.clip_y(max_value=self.clip_acc_y)
        if self.clip_gyro_y is not None:
            gyro = gyro.clip_y(max_value=self.clip_gyro_y)
        if self.clip_acc_z is not None:
            acc = acc.clip_z(max_value=self.clip_acc_z)
        if self.clip_gyro_z is not None:
            gyro = gyro.clip_z(max_value=self.clip_gyro_z)

        return ImuStream(acc=acc, gyro=gyro)
