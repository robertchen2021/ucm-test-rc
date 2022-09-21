from .interfaces import AbstractSensorPreprocessor
from .combined import SensorPreprocessorCombined
from nauto_zoo import ModelInput, TooShortSensorStreamError, MalformedModelInputError, DoNotWantToProduceJudgementError
import numpy as np
from nauto_datasets.core.sensors import ImuStream
from typing import List, Dict
import pandas as pd
from scipy import interpolate

class SensorMetadataPreprocessor(AbstractSensorPreprocessor):
    DISTANCE_NORM_FACTOR = 100
    TTC_NORM_FACTOR = 10
    SPEED_NORM_FACTOR = 50
    LOOKBACK_S = 8
    OBSERVATION_WINDOW = 30
    ALL_METADATA_COLUMNS = ['distance_estimate', 'score_looking_down', 'score_looking_up', 'score_looking_left',
                            'score_looking_right', 'score_cell_phone', 'score_smoking', 'score_holding_object',
                            'score_eyes_closed', 'score_no_face', 'ttc', 'speed']
    def __init__(
            self,
            observation_window: float = OBSERVATION_WINDOW,
            lookback_s: float = LOOKBACK_S,
            distance_norm_factor: float = DISTANCE_NORM_FACTOR,
            ttc_norm_factor: float = TTC_NORM_FACTOR,
            speed_norm_factor: float = SPEED_NORM_FACTOR,
            metadata_columns: List[str] = ALL_METADATA_COLUMNS):
        self._observation_window = observation_window
        self._lookback_s = lookback_s
        self._distance_norm_factor = distance_norm_factor
        self._ttc_norm_factor = ttc_norm_factor
        self._speed_norm_factor = speed_norm_factor
        self._metadata_columns = metadata_columns

        self._sensor_preprocessor = SensorPreprocessorCombined()

    # def preprocess_sensor_files(self, sensor_files: List[str]) -> ModelInput:
    def preprocess_sensor_files(self, sensor_files: List[str], metadata: Dict = None):
        com_recordings: CombinedRecording = self._sensor_preprocessor.preprocess_sensor_files(sensor_files)

        df_acc = getattr(com_recordings, 'rt_oriented_acc').stream._to_df().drop('system_ms', axis=1).rename(
            columns={'sensor_ns': 'timestamp_utc_ns'}).drop_duplicates(subset=['timestamp_utc_ns']).reset_index(
            drop=True)

        df_acc = df_acc.set_index('timestamp_utc_ns', drop=False, inplace=False)
        df_acc.sort_index(inplace=True)
        df_acc = df_acc.dropna()
        max_idx = (df_acc.x ** 2 + df_acc.y ** 2 + df_acc.z ** 2).values.argmax()
        max_ns = df_acc.timestamp_utc_ns.values[max_idx]

        peak_ns = max_ns
        t_ref = np.arange(int(peak_ns - (self._lookback_s * 1e9)), peak_ns, 200 * 1e6, dtype=np.uint64)
        ref_df = pd.DataFrame(t_ref, columns=['timestamp_utc_ns'])

        dist_df = getattr(com_recordings, 'dist_multilabel').stream._to_df().drop('system_ms', axis=1).rename(
            columns={'sensor_ns': 'timestamp_utc_ns'}).drop_duplicates(subset=['timestamp_utc_ns']).reset_index(
            drop=True)
        dist_df = dist_df.set_index('timestamp_utc_ns', drop=False, inplace=False)
        dist_df.sort_index(inplace=True)

        tail_df = getattr(com_recordings, 'tailgating').stream._to_df().drop('system_ms', axis=1).rename(
            columns={'sensor_ns': 'timestamp_utc_ns'}).drop_duplicates(subset=['timestamp_utc_ns']).reset_index(
            drop=True)
        tail_df = tail_df.set_index('timestamp_utc_ns', drop=False, inplace=False)
        tail_df.sort_index(inplace=True)

        gps_df = getattr(com_recordings, 'gps').stream._to_df().drop('system_ms', axis=1).rename(
            columns={'sensor_ns': 'timestamp_utc_ns'}).drop_duplicates(subset=['timestamp_utc_ns']).reset_index(
            drop=True)
        gps_df = gps_df.set_index('timestamp_utc_ns', drop=False, inplace=False)
        gps_df.sort_index(inplace=True)

        fcw_df = getattr(com_recordings, 'fcw').stream._to_df().drop('system_ms', axis=1).rename(
            columns={'sensor_ns': 'timestamp_utc_ns'}).drop_duplicates(subset=['timestamp_utc_ns']).reset_index(
            drop=True)
        fcw_df.set_index('timestamp_utc_ns', drop=False, inplace=False)
        fcw_df.sort_index(inplace=True)

        tail_df.drop(columns=['score', 'front_box_index'], inplace=True)

        # print('Stream lengths - ref: %d, gps: %d, dist: %d, tail: %d, fcw: %d' % (
        # ref_df.shape[0], gps_df.shape[0], dist_df.shape[0], tail_df.shape[0], fcw_df.shape[0]))
        if dist_df.timestamp_utc_ns.values[0] > t_ref[0]:
            raise DoNotWantToProduceJudgementError("Not enough dist data")

        if tail_df.shape[0] > 1:
            get_from = interpolate.interp1d(tail_df.timestamp_utc_ns.values, tail_df.distance_estimate.values,
                                            fill_value='extrapolate', kind='nearest')
            ref_df['distance_estimate'] = get_from(ref_df.timestamp_utc_ns.values)
        else:
            ref_df['distance_estimate'] = 0 * ref_df.timestamp_utc_ns.values + 100
        dist_labels = ['score_looking_down', 'score_looking_up', 'score_looking_left', 'score_looking_right',
                       'score_cell_phone', 'score_smoking', 'score_holding_object', 'score_eyes_closed',
                       'score_no_face']
        for dist_label in dist_labels:
            get_from = interpolate.interp1d(dist_df.timestamp_utc_ns.values, dist_df[dist_label].values,
                                            fill_value='extrapolate', kind='nearest')
            ref_df[dist_label] = get_from(ref_df.timestamp_utc_ns.values)

        if fcw_df.shape[0] > 1:
            get_from = interpolate.interp1d(fcw_df.timestamp_utc_ns.values, fcw_df.ttc.values, fill_value='extrapolate',
                                            kind='nearest')
            ttc_vals = get_from(ref_df.timestamp_utc_ns.values)
            invalid_locations = np.where((ttc_vals < 0.5) | (ttc_vals > 6))[0]  # force all <0.5 and >6 s to 10
            ttc_vals[invalid_locations] = 10
            ref_df['ttc'] = ttc_vals
        else:
            ref_df['ttc'] = 0 * ref_df.timestamp_utc_ns.values + 10
        get_from = interpolate.interp1d(gps_df.timestamp_utc_ns.values, gps_df.speed.values, fill_value='extrapolate',
                                        kind='nearest')
        ref_df['speed'] = get_from(ref_df.timestamp_utc_ns.values)
        sample_input = ref_df.loc[:, self._metadata_columns].values
        sample_input[:, 0] = sample_input[:, 0] / self._distance_norm_factor  # Normalize distance
        sample_input[:, 10] = sample_input[:, 10] / self._ttc_norm_factor  # Normalize ttc value
        sample_input[:, 11] = sample_input[:, 11] / self._speed_norm_factor  # Normalize speed

        return np.expand_dims(np.expand_dims(sample_input[:self._observation_window, :], axis=0), axis=-1).astype(np.float32)
