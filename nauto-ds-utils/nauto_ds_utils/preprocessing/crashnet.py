from nauto_ds_utils.preprocessing.imu import SensorPreprocessorImu
from nauto_datasets.core.sensors import ImuStream
from typing import List, Any
import numpy as np


class CrashnetSensorPreprocessor():
    DEFAULT_WINDOW_SIZE = 2000
    DEFAULT_MAX_NS_BETWEEN_MEASUREMENTS = 1e9
    DEFAULT_GYRO_SCALING_FACTOR = 25
    DEFAULT_GYRO_CLIP = 220.1702
    DEFAULT_USE_ORIENTED_STREAMS = False

    def __init__(
            self,
            window_size: int = DEFAULT_WINDOW_SIZE,
            max_ns_between_measurements: float = DEFAULT_MAX_NS_BETWEEN_MEASUREMENTS,
            gyro_scaling_factor: float = DEFAULT_GYRO_SCALING_FACTOR,
            gyro_clip: float = DEFAULT_GYRO_CLIP,
            use_oriented: bool = DEFAULT_USE_ORIENTED_STREAMS,
            imu_preprocessor: Any = SensorPreprocessorImu()
    ):
        self._window_size = window_size
        self._max_ns_between_measurements = max_ns_between_measurements
        self._gyro_scaling_factor = gyro_scaling_factor
        self._gyro_clip = gyro_clip
        self._use_oriented = use_oriented
        self._imu_preprocessor = imu_preprocessor

    def sort_xyz_stream(self, xyzstream, sort_idx):
        xyzstream.sensor_ns[:] = xyzstream.sensor_ns[sort_idx]
        xyzstream.x[:] = xyzstream.x[sort_idx]
        xyzstream.y[:] = xyzstream.y[sort_idx]
        xyzstream.z[:] = xyzstream.z[sort_idx]

    def preprocess_sensor_files(self, sensor_files: List[str]) -> np.array:
        imu_stream: ImuStream = self._imu_preprocessor.preprocess_sensor_files(sensor_files, oriented_streams=self._use_oriented)

        self.sort_xyz_stream(
            imu_stream.acc, np.argsort(imu_stream.acc.sensor_ns))
        self.sort_xyz_stream(
            imu_stream.gyro, np.argsort(imu_stream.gyro.sensor_ns))

        if imu_stream.acc.x.shape[0] < self._window_size:
            raise ValueError(
                "Stream acc_x is too short - should have length at least %d but has length %d" % (
                    imu_stream.acc.x.shape[0], self._window_size))
        if (np.diff(imu_stream.acc.sensor_ns) > self._max_ns_between_measurements).any():
            raise ValueError(
                "`sensor_ns` looks fishy, some measurements seem to be missing")

        # find the idx wrt the maximum peak
        max_idx = imu_stream.acc.find_peak()

        start_idx = int(max_idx - self._window_size / 2)
        end_idx = int(max_idx + self._window_size / 2)

        if start_idx < 0:
            # left index out of bound, choose the very first WINDOW_SIZE samples
            start_idx = 0
            end_idx = self._window_size
        elif end_idx > imu_stream.acc.x.shape[0] - 1:
            # right index out of bound, choose the very last WINDOW_SIZE samples
            start_idx = -self._window_size
        else:
            pass

        try:
            data = np.vstack((
                imu_stream.acc.x[start_idx:end_idx],
                imu_stream.acc.y[start_idx:end_idx],
                imu_stream.acc.z[start_idx:end_idx],
                np.clip(
                    imu_stream.gyro.x[start_idx:end_idx] *
                    self._gyro_scaling_factor,
                    a_min=-self._gyro_clip,
                    a_max=self._gyro_clip
                ),
                np.clip(
                    imu_stream.gyro.y[start_idx:end_idx] *
                    self._gyro_scaling_factor,
                    a_min=-self._gyro_clip,
                    a_max=self._gyro_clip
                ),
                np.clip(
                    imu_stream.gyro.z[start_idx:end_idx] *
                    self._gyro_scaling_factor,
                    a_min=-self._gyro_clip,
                    a_max=self._gyro_clip
                )
            ))
        except:
            raise ValueError(
                "`acc` and `gyro` stream lengths do not match")
        return data.reshape(1, 6, -1, 1).transpose((0, 2, 1, 3))
