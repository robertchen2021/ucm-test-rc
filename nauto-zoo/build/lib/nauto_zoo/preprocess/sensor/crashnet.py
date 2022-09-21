from .interfaces import AbstractSensorPreprocessor
from .imu import SensorPreprocessorImu
from nauto_zoo import ModelInput, TooShortSensorStreamError, MalformedModelInputError, DoNotWantToProduceJudgementError
import numpy as np
from nauto_datasets.core.sensors import ImuStream
from typing import List, Dict
import logging

class CrashnetSensorPreprocessor(AbstractSensorPreprocessor):
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
            use_oriented: bool = DEFAULT_USE_ORIENTED_STREAMS):
        self._window_size = window_size
        self._max_ns_between_measurements = max_ns_between_measurements
        self._gyro_scaling_factor = gyro_scaling_factor
        self._gyro_clip = gyro_clip
        self._use_oriented = use_oriented
        self._imu_preprocessor = SensorPreprocessorImu()
        self._logger = logging.getLogger()
        
    def sort_xyz_stream(self, xyzstream, sort_idx):
        xyzstream.sensor_ns[:] = xyzstream.sensor_ns[sort_idx]
        xyzstream.x[:] = xyzstream.x[sort_idx]
        xyzstream.y[:] = xyzstream.y[sort_idx]
        xyzstream.z[:] = xyzstream.z[sort_idx]

    def preprocess_sensor_files(self, sensor_files: List[str], metadata: Dict = None) -> ModelInput:
        # This first uses the combined preprocessor to deserialize protobuf and fill in missing raw/oriented stream using the rt_device_orientation
        # Then IMU preprocessor sorts the strams and clips gyro value
        imu_stream: ImuStream = self._imu_preprocessor.preprocess_sensor_files(sensor_files, oriented_streams=self._use_oriented)

        if imu_stream.acc.x.shape[0] < self._window_size:
            raise DoNotWantToProduceJudgementError(
                "Stream acc_x is too short - should have length at least %d but has length %d" % (
                imu_stream.acc.x.shape[0], self._window_size))
        if (np.diff(imu_stream.acc.sensor_ns) > self._max_ns_between_measurements).any():
            raise DoNotWantToProduceJudgementError("`sensor_ns` looks fishy, some measurements seem to be missing")

        if metadata["type"] == "crashnet":
            peak_sensor_ns = int(metadata['params']['crashnet_data']['computed_peak_sensor_ns'])
        elif metadata["type"] == "severe-g-event":
            peak_sensor_ns = int(metadata['params']['maneuver_data']['peak_time_ns'])
        else:
            peak_sensor_ns = None

        if peak_sensor_ns is not None:
            self._logger.info("peak_sensor_ns from event type %s: %d"%(metadata["type"], peak_sensor_ns))
            max_idx = np.argmin(np.abs(imu_stream.acc.sensor_ns - peak_sensor_ns))
        else:
            # find the idx wrt the maximum peak
            self._logger.info("peak_sensor_ns not found, computing acc peak")
            max_idx = imu_stream.acc.find_peak()

        start_idx = int(max_idx - self._window_size / 2)
        end_idx = int(max_idx + self._window_size / 2)

        if start_idx < 0:
            self._logger.info("left index out of bound, selecting the very first WINDOW_SIZE samples")
            start_idx = 0
            end_idx = self._window_size
        elif end_idx > imu_stream.acc.x.shape[0] - 1:
            self._logger.info("right index out of bound, selecting the very last WINDOW_SIZE samples")
            start_idx = -self._window_size
        else:
            self._logger.info("start and end idx within bound")

        try:
            data = np.vstack((
                imu_stream.acc.x[start_idx:end_idx],
                imu_stream.acc.y[start_idx:end_idx],
                imu_stream.acc.z[start_idx:end_idx],
                np.clip(
                    imu_stream.gyro.x[start_idx:end_idx] * self._gyro_scaling_factor,
                    a_min=-self._gyro_clip,
                    a_max=self._gyro_clip
                ),
                np.clip(
                    imu_stream.gyro.y[start_idx:end_idx] * self._gyro_scaling_factor,
                    a_min=-self._gyro_clip,
                    a_max=self._gyro_clip
                ),
                np.clip(
                    imu_stream.gyro.z[start_idx:end_idx] * self._gyro_scaling_factor,
                    a_min=-self._gyro_clip,
                    a_max=self._gyro_clip
                )
            ))
        except:
            raise DoNotWantToProduceJudgementError("`acc` and `gyro` stream lengths do not match")
        return data.reshape(1, 6, -1, 1).transpose((0, 2, 1, 3))
