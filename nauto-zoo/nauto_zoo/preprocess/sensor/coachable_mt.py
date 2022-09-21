from .interfaces import AbstractSensorPreprocessor
from .combined import SensorPreprocessorCombined
from nauto_zoo import ModelInput, TooShortSensorStreamError, MalformedModelInputError, DoNotWantToProduceJudgementError
import numpy as np
from nauto_datasets.core.sensors import CombinedRecording, ImuStream
from typing import List, Dict
import pandas as pd
from scipy import interpolate


"""
only modified to have 10 streams of preprocessor
TODO: apply downsampling & interpolation, padding, filtering
"""


class SensorCoachablePreprocessor_mt(AbstractSensorPreprocessor):
    N_SEC_BEFORE_PEAK = 5.0
    N_SEC_AFTER_PEAK = 3.0
    SAMPLE_RATE = 10.0
    PAD_VALUE = -1.0
    STREAMS_TO_EXTRACT = {'rt_oriented_acc': [1, 2, 3],
                          'rt_oriented_gyro': [1, 2, 3],
                          'gps': [4],
                          'dist_multilabel': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          'tailgating': [2],
                          'fcw': [1],
                          }  # TODO: change this to the stream name instead
    NORMALIZATION_FACTORS = {'rt_oriented_acc': 10.0,
                             'rt_oriented_gyro': 0.02,
                             'gps': 40,
                             'tailgating': 100,
                             'fcw': 10,
                             }

    def __init__(
            self,
            sample_rate: float = SAMPLE_RATE,
            n_sec_before_peak: float = N_SEC_BEFORE_PEAK,
            n_sec_after_peak: float = N_SEC_AFTER_PEAK,
            pad_value: float = PAD_VALUE,
            normalization_factors: Dict = NORMALIZATION_FACTORS,
            streams_to_extract: Dict = STREAMS_TO_EXTRACT):

        # get the data trimming and padding parameters
        self._n_sec_before_peak = n_sec_before_peak
        self._n_sec_after_peak = n_sec_after_peak
        self._pad_value = pad_value

        # get the data down-sampling and up-sampling parameters
        self._sample_rate = sample_rate
        self._dt = 1 / self._sample_rate
        self._window_len = int((self._n_sec_before_peak + self._n_sec_after_peak) * self._sample_rate) + 1

        # get the normalization factors
        self._streams_to_extract = streams_to_extract
        self._normalization_factors = normalization_factors

        self._sensor_preprocessor = SensorPreprocessorCombined()

    def _apply_filter(self, data_in, param_key):
        """
        TODO: edge case handling. For now, simply return the input data
        """
        # # create filter
        # if self.filter_params[param_key]['filter_type'] == 'butter_lowpass':
        #     b, a = butter_lowpass(cutoff=self.filter_params[param_key]['filter_cof'],
        #                           fs=200,
        #                           order=self.filter_params[param_key]['filter_order'])
        # else:
        #     raise ValueError(f"The specified filter_type '{self.filter_params[param_key]['filter_type']}' is "
        #                      f"not supported.")
        #
        # # apply filter
        # if self.filter_params[param_key]['filter_apply_func'] == 'filtfilt':
        #     data_out = sp.signal.filtfilt(b, a, data_in, axis=0)
        #     return data_out
        # else:
        #     raise ValueError(f"The specified filter_apply_func '{self.filtering[param_key]['filter_apply_func']}' is "
        #                      f"not supported.")
        return data_in

    def _normalize_data(self, data_in, param_key: str):
        return data_in / self._normalization_factors[param_key]

    def _interpolate_data(self, t_in, data_in, t_out):
        if t_in.shape[0] < 2:
            data_out = np.zeros((t_out.shape[0], data_in.shape[1]))
        else:
            interpolator = interpolate.interp1d(t_in, data_in, axis=0, bounds_error=False, fill_value=self._pad_value,
                                                kind='linear')
            data_out = interpolator(t_out).reshape((t_out.shape[0], data_in.shape[1]))
        return data_out

    # def preprocess_sensor_files(self, sensor_files: List[str]) -> ModelInput:
    def preprocess_sensor_files(self, sensor_files: List[str], metadata: Dict = None):
        com_recordings: CombinedRecording = self._sensor_preprocessor.preprocess_sensor_files(sensor_files)

        # get acc data (will use the rt_oriented_acc)
        acc_df = (getattr(com_recordings, 'rt_oriented_acc').stream._to_df()
                  .drop('system_ms', axis=1)
                  .rename(columns={'sensor_ns': 'timestamp_utc_ns'})
                  .drop_duplicates(subset=['timestamp_utc_ns'])
                  .reset_index(drop=True))
        acc_df = acc_df.set_index('timestamp_utc_ns', drop=False, inplace=False)
        acc_df.sort_index(inplace=True)
        acc_df = acc_df.dropna()

        # calculate the peak timestamp from the acc data
        max_idx = (acc_df.x ** 2 + acc_df.y ** 2 + acc_df.z ** 2).values.argmax()
        max_ns = acc_df.timestamp_utc_ns.values[max_idx]
        peak_ns = max_ns

        # construct the timestamps to interpolate to, based on the peak timestamp
        t_ref = np.arange(peak_ns - self._n_sec_before_peak*1e9, peak_ns + (self._n_sec_after_peak+self._dt)*1e9,
                          self._dt*1e9, dtype=np.uint64)

        data_out = []
        for stream_name, stream_idx_to_use in self._streams_to_extract.items():
            data_raw = (getattr(com_recordings, stream_name).stream._to_df()
                        .drop('system_ms', axis=1)
                        .rename(columns={'sensor_ns': 'timestamp_utc_ns'})
                        .drop_duplicates(subset=['timestamp_utc_ns'])
                        .reset_index(drop=True))
            t_in = data_raw.iloc[:, 0].values
            data_to_use = data_raw.iloc[:, stream_idx_to_use].values

            # # apply filter
            # turned off for now -- the previous code needs edge case handling
            # data_to_use = self._apply_filter(data_in=data_to_use, param_key=channel)

            # apply normalization
            if stream_name in self._normalization_factors:
                data_to_use = self._normalize_data(data_in=data_to_use, param_key=stream_name)

            # apply interpolation
            data_to_use = self._interpolate_data(t_in=t_in, data_in=data_to_use, t_out=t_ref)

            # expand the dimension for batch size
            data_to_use = np.expand_dims(data_to_use, axis=0)
            data_out.append(data_to_use)

        # return as a list
        # Note: need to encapsulate in one more layer of list,
        # to ensure the keras_v2 model considers the list of 6 items as 1 batch, instead of a batch of 6
        return [data_out]
