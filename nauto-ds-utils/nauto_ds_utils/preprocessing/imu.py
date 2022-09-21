from nauto_ds_utils.utils.sensor import get_combined_recording
from nauto_datasets.core.sensors import ImuStream, XYZStream, CombinedRecording
from typing import List, NamedTuple


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
            gyro = gyro._replace(
                x=gyro.x * 4.0, y=gyro.y * 4.0, z=gyro.z * 4.0)

        return OrientedImuStream(acc=recording.oriented_acc.stream, gyro=gyro)


class RTOrientedImuStream(NamedTuple):
    acc: XYZStream
    gyro: XYZStream

    @staticmethod
    def from_recording(recording: CombinedRecording) -> "RTOrientedImuStream":
        if recording.rt_oriented_acc.stream._is_empty() or recording.rt_oriented_gyro.stream._is_empty():
            if recording.oriented_acc.stream._is_empty() or recording.oriented_gyro.stream._is_empty():
                raise ValueError("Oriented streams are empty!")
            else:
                return RTOrientedImuStream(acc=recording.oriented_acc.stream, gyro=recording.oriented_gyro.stream)
        return RTOrientedImuStream(acc=recording.rt_oriented_acc.stream, gyro=recording.rt_oriented_gyro.stream)


class RTImuStream(NamedTuple):
    acc: XYZStream
    gyro: XYZStream

    @staticmethod
    def from_recording(recording: CombinedRecording) -> "RTImuStream":
        if recording.rt_acc.stream._is_empty() or recording.rt_gyro.stream._is_empty():
            if recording.acc.stream._is_empty() or recording.gyro.stream._is_empty():
                raise ValueError("Streams are empty!")
            else:
                RTImuStream(acc=recording.acc.stream,
                            gyro=recording.gyro.stream)

        return RTImuStream(acc=recording.rt_acc.stream, gyro=recording.rt_gyro.stream)


class SensorPreprocessorImu(object):
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
        self.max_length = max_length
        self.cut_beginning = cut_beginning
        self.clip_acc_x = clip_acc_x
        self.clip_gyro_x = clip_gyro_x
        self.clip_acc_y = clip_acc_y
        self.clip_gyro_y = clip_gyro_y
        self.clip_acc_z = clip_acc_z
        self.clip_gyro_z = clip_gyro_z

    def preprocess_sensor_files(self,
                                sensor_files: List[str],
                                oriented_streams: bool = False) -> ImuStream:
        if oriented_streams:
            imu_stream = RTOrientedImuStream.from_recording(
                get_combined_recording(sensor_files))
        else:
            imu_stream = RTImuStream.from_recording(
                get_combined_recording(sensor_files))

        acc = imu_stream.acc
        gyro = imu_stream.gyro
        if self.max_length is not None:
            acc = acc.cut(max_length=self.max_length,
                          cut_beginning=self.cut_beginning),
            gyro = gyro.cut(max_length=self.max_length,
                            cut_beginning=self.cut_beginning),
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