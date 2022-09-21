"""
@author: mohammad.billah@nauto.com
This class processes sensor protobuf data for safer ucm
This uses nauto-datasets to deserialize protobufs into com_rec
and then performs similar pre-processing to create the multi-stream input for safer model
Additionally returns some info related to the event and sensor file for debugging purposes
Dependency: nauto-datasets
"""

from .interfaces import AbstractSensorPreprocessor
from .combined import SensorPreprocessorCombined
from nauto_zoo import ModelInput, TooShortSensorStreamError, MalformedModelInputError, DoNotWantToProduceJudgementError
import numpy as np
from ast import literal_eval
from nauto_datasets.core.sensors import ImuStream
from typing import List, Dict
import pandas as pd
from scipy import interpolate
from collections import OrderedDict

class SaferSensorPreprocessor(AbstractSensorPreprocessor):
    SPEED_NORM_FACTOR = 50.0
    ACC_NORM_FACTOR = 10.0
    FPS = 5
    LOOKBACK_S = 8
    PREDICTION_DELTA_S = 2
    OBSERVATION_WINDOW = 30
    DISTRACTION_COLUMNS = ['score_looking_down', 'score_looking_up', 'score_looking_left',
                            'score_looking_right', 'score_cell_phone', 'score_smoking', 'score_holding_object',
                            'score_eyes_closed', 'score_no_face']

    BBOX_DICT = OrderedDict([('vehicle', 1), ('pedestrian', 4), ('bicyclist', 15), ('motorcyclist', 16), ('stopline_left', 13), ('stopline_right', 14), ('other_vehicle', 5)])
    def __init__(
            self,
            observation_window: float = OBSERVATION_WINDOW,
            lookback_s: float = LOOKBACK_S,
            prediction_delta_s: float = PREDICTION_DELTA_S,
            acc_norm_factor: float = ACC_NORM_FACTOR,
            speed_norm_factor: float = SPEED_NORM_FACTOR,
            fps: float = FPS,
            distraction_columns: List[str] = DISTRACTION_COLUMNS,
            bbox_dict: Dict = BBOX_DICT):
        self._observation_window = observation_window
        self._lookback_s = lookback_s
        self._prediction_delta_s = prediction_delta_s
        self._acc_norm_factor = acc_norm_factor
        self._speed_norm_factor = speed_norm_factor
        self._fps = fps
        self._distraction_columns = distraction_columns
        self._bbox_dict = bbox_dict
        self._sensor_preprocessor = SensorPreprocessorCombined()

    def extract_df_from_com_rec(self, com_rec, stream_name):
        df = getattr(com_rec, stream_name).stream._to_df().drop('system_ms', axis=1).rename(
            columns={'sensor_ns': 'timestamp_utc_ns'}).drop_duplicates(subset=['timestamp_utc_ns']).reset_index(
            drop=True)
        df = df.set_index('timestamp_utc_ns', drop=False, inplace=False)
        df.sort_index(inplace=True)
        return df

    def extract_bbox(self, bbox_array, object_type):
        column_idx = np.where(bbox_array[4, :].astype(int) == object_type)[0]
        if column_idx.shape[0] > 0:
            column_idx = column_idx[0]
            return bbox_array[:4, column_idx].tolist()
        else:
            return [0.0] * 4

    def preprocess_sensor_files(self, sensor_files: List[str], metadata: Dict = None):
        video_start_ns = min([int(item['params']['sensor_start']) for item in metadata['media'] if item['type'].find('video') != -1])
        com_recordings: CombinedRecording = self._sensor_preprocessor.preprocess_sensor_files(sensor_files)

        # Extract the required streams as df
        df_acc = self.extract_df_from_com_rec(com_recordings, stream_name='rt_oriented_acc')
        dist_df = self.extract_df_from_com_rec(com_recordings, stream_name='dist_multilabel')
        gps_df = self.extract_df_from_com_rec(com_recordings, stream_name='gps')
        bbox_df = self.extract_df_from_com_rec(com_recordings, stream_name='mcod_bounding_boxes')

        # Compute the peak
        df_acc = df_acc.dropna()
        max_idx = (df_acc.x ** 2 + df_acc.y ** 2 + df_acc.z ** 2).values.argmax()
        max_ns = df_acc.timestamp_utc_ns.values[max_idx]
        peak_ns = max_ns

        # Create ref time-axis for input
        t_ref = np.arange(int(peak_ns - (self._lookback_s * 1e9)), int(peak_ns - (self._prediction_delta_s * 1e9)), (1000/self._fps) * 1e6, dtype=np.uint64)


        # interpolate dist stream
        f = interpolate.interp1d(dist_df.timestamp_utc_ns.values, dist_df[self._distraction_columns].values, axis=0,
                                 fill_value='extrapolate', kind='nearest')
        dist_input = np.expand_dims(np.expand_dims(f(t_ref), axis=-1), axis=0)

        # interpolate bbox streams
        # first parse the bboxes
        bbox_df['bbox'] = bbox_df['bounding_box'].map(
            lambda x: np.array([getattr(x, 'left').tolist(), getattr(x, 'top').tolist(), getattr(x, 'right').tolist(),
                                getattr(x, 'bottom').tolist(), getattr(x, 'objectType').tolist(),
                                getattr(x, 'score').tolist()]))
        bbox_input = {}
        for key, val in self._bbox_dict.items():
            if bbox_df.shape[0] > 0:
                bbox_df[key] = bbox_df.apply(lambda row: self.extract_bbox(row['bbox'], object_type=val), axis=1)
                f = interpolate.interp1d(bbox_df.timestamp_utc_ns.values, np.stack(bbox_df[key].values), axis=0,
                                 fill_value='extrapolate', kind='nearest')
                bbox_input[key] = np.expand_dims(np.expand_dims(f(t_ref), axis=-1), axis=0)
            else:
                bbox_input[key] = np.zeros((1, t_ref.shape[0], 4, 1))

        # interpolate gps speed
        f = interpolate.interp1d(gps_df.timestamp_utc_ns.values, gps_df[['speed']].values, axis=0,
                                 fill_value='extrapolate', kind='nearest')
        speed_input = np.expand_dims(np.expand_dims(f(t_ref), axis=-1), axis=0)


        # interpolate acc stream
        half_len = int(200 / (2 * self._fps))
        acc_array = []
        for ii in range(t_ref.shape[0]):
            iidx = np.nanargmin(np.abs(df_acc.timestamp_utc_ns.values - t_ref[ii]))
            acc_avg_around_iidx = np.expand_dims(np.array([np.mean(df_acc.x.values[iidx - half_len:iidx + half_len]),
                                                           np.mean(df_acc.y.values[iidx - half_len:iidx + half_len]),
                                                           np.mean(df_acc.z.values[iidx - half_len:iidx + half_len])]),
                                                 axis=-1)
            acc_array.append(acc_avg_around_iidx)

        info = {
            'input_start_s': (float(t_ref[0]) - float(video_start_ns))/1e9,
            'input_end_s': (float(t_ref[-1]) - float(video_start_ns)) / 1e9,
        }
        bbox_count = [np.count_nonzero(item[0,:,2,0]-item[0,:,0,0]) for item in list(bbox_input.values())]
        for idx, key in enumerate(list(self._bbox_dict.keys())):
            info[key+'_count'] = bbox_count[idx]

        safer_v2_input = [dist_input] + list(bbox_input.values()) + [speed_input / self._speed_norm_factor,
                                                                     np.expand_dims(np.array(acc_array), axis=0) / self._acc_norm_factor]
        return safer_v2_input, info
