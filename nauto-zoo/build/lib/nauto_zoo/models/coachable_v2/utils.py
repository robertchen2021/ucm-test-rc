"""
This file contains some util tools that are used on the processing functions
com_rec.rt_* is used for config_version >= 3.4
com_rec.* is used for config_version < 3.4
TODO: remove com_rec.* checking after all devices get updated to 3.4
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from nauto_datasets.core.sensors import CombinedRecording, Recording
from nauto_datasets.utils import protobuf
from nauto_zoo import IncompleteInputMediaError, MissingSensorStreamError
from sensor import sensor_pb2


def interp_df(df_in: pd.DataFrame,
              x_col_name: str,
              x_new_vals: np.ndarray) -> pd.DataFrame:
    if x_col_name not in df_in.columns:
        raise ValueError(f'x_col_name: "{x_col_name}" not in the target dataframe')

    df_out = pd.DataFrame()
    df_out[x_col_name] = x_new_vals

    for col in df_in.columns:
        if col != x_col_name:
            df_out[col] = np.interp(x_new_vals, df_in[x_col_name], df_in[col])

    return df_out


def get_consecutive_time_segments(times, indices):
    """
    Find consecutive time segments based on indices
    :param times:
    :param indices:
    :return:
    """

    indices = sorted(indices)
    # initialize first segment with element
    segments = []
    idx_curr = indices[0]
    t_start = times[idx_curr]
    curr_segment_indices = [idx_curr]
    segment_indices = []

    for k in range(1, len(indices)):
        idx_new = indices[k]
        if idx_new - idx_curr == 1:
            # update current segment
            curr_segment_indices.append(idx_new)
        else:
            # close segment
            segments.append((t_start, times[idx_curr]))
            segment_indices.append(curr_segment_indices)
            # start new segment
            t_start = times[idx_new]
            curr_segment_indices = [idx_new]
        idx_curr = idx_new
    # append last segment
    segments.append((t_start, times[idx_curr]))
    segment_indices.append(curr_segment_indices)
    return segments, segment_indices


def sec_to_sensor_ns(tarr: List[np.ndarray],
                     offset_ns: int = 0) -> List[int]:
    return (np.int64(np.array(tarr) * 1e9 + offset_ns)).tolist()


##############################
# Protobuf Utils
##############################
def get_combined_recording(paths: List[str]) -> Optional['CombinedRecording']:
    """
    :param paths:
    :return:
    """

    if None in paths:
        logging.error('None path in sensor paths')
        return None

    try:
        gzip_files = [Path(p) for p in paths]
        recordings = []
        for file_path in gzip_files:
            rec_pb = protobuf.parse_message_from_gzipped_file(sensor_pb2.Recording, file_path)
            recordings.append(Recording.from_pb(rec_pb))
        com_rec = CombinedRecording.from_recordings(recordings)
        return com_rec

    except Exception:
        logging.exception('Could not read sensor data')
        return None


def get_sensors_from_combined_recordings(com_rec: 'CombinedRecording') -> Optional[Dict[str, Any]]:
    """
    :param com_rec:
    :return:
    """
    # TODO: add other sensors, check for empty sensors data

    if not com_rec:
        raise IncompleteInputMediaError('No CombinedRecording given in the model_input.')

    if not hasattr(com_rec, 'oriented_acc'):
        raise MissingSensorStreamError(message='Sensor com_rec is missing oriented_acc streams.',
                                       missing_streams=['oriented_acc'])

    acc_orig = com_rec.rt_oriented_acc.stream._asdict()
    if 'sensor_ns' not in acc_orig.keys() or len(acc_orig['sensor_ns']) < 1:
        acc_orig = com_rec.oriented_acc.stream._asdict()

    if 'sensor_ns' not in acc_orig.keys() or len(acc_orig['sensor_ns']) < 1:
        raise MissingSensorStreamError(message='Sensor com_rec is missing oriented_acc streams.',
                                       missing_streams=['oriented_acc'])

    # select fields
    imu = dict()
    imu['sensor_ns'] = acc_orig['sensor_ns'].copy()

    acc_fields = ['x', 'y', 'z']
    for key in acc_fields:
        imu[f'acc_{key}'] = acc_orig[key].copy()

    # sort data by sensor_ns
    sort_idx = np.argsort(imu['sensor_ns'])
    for key in imu.keys():
        imu[key] = imu[key][sort_idx]

    return imu


def com_rec_to_df_sensors(com_rec: 'CombinedRecording') -> pd.DataFrame:
    """
    Parse combined records for sensors and store as pandas dataframe
    :param com_rec:
    :return:
    """

    # extract the signals
    acc = com_rec.rt_oriented_acc.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)
    if acc.empty:
        acc = com_rec.oriented_acc.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)

    gyro = com_rec.rt_oriented_gyro.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)
    if gyro.empty:
        gyro = com_rec.oriented_gyro.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)

    gps = com_rec.gps.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)

    # throw exception is any stream is missing
    missing_streams = []
    if acc.empty:
        missing_streams.append('oriented_acc')
    if gyro.empty:
        missing_streams.append('oriented_gyro')
    if gps.empty:
        missing_streams.append('gps')
    if len(missing_streams) > 0:
        raise MissingSensorStreamError(message=f"Sensor com_rec is missing {', '.join(missing_streams)} streams.",
                                       missing_streams=missing_streams)

    # rename columns
    acc.rename(columns={c: f'acc_{c}' for c in ['x', 'y', 'z']}, inplace=True)
    gyro.rename(columns={c: f'gyr_{c}' for c in ['x', 'y', 'z']}, inplace=True)

    # remove system_ms column
    acc.drop('system_ms', axis=1, inplace=True)
    gyro.drop('system_ms', axis=1, inplace=True)

    # interpolate GPS data to sensor_ns
    ngps = pd.DataFrame()
    ngps['sensor_ns'] = acc.sensor_ns
    for col in ['longitude', 'latitude', 'speed']:
        ngps[col] = np.interp(acc.sensor_ns, gps.sensor_ns, gps[col])

    # merge different sensors based on sensor_ns
    # !!! Check if this is proper way to do it
    join_col = 'sensor_ns'
    sensors = (acc
               .merge(gyro, on=join_col)
               .merge(ngps, on=join_col)
               )
    return sensors


def com_rec_to_df_video_scores(com_rec: 'CombinedRecording') -> pd.DataFrame:
    """
    Parse combined records for video scores and store as pandas dataframe
    :param com_rec:
    :return:
    """

    # extract tailgating scores
    tg = com_rec.tailgating.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)
    tg['score_tailgating'] = tg['score']
    tg.drop(columns=['system_ms', 'score'], axis=1, inplace=True)

    # extract distraction scores
    di = com_rec.dist_multilabel.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)
    di.drop(columns=['system_ms'], axis=1, inplace=True)

    # merge
    return tg.merge(di, on='sensor_ns', how='outer')


def com_rec_to_df(com_rec: 'CombinedRecording') -> pd.DataFrame:
    """
    Parse combined records for sensors and video scores, and store as pandas dataframe
    :param com_rec:
    :return:
    """

    # extract the signals from sensors
    acc = com_rec.rt_oriented_acc.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)
    if acc.empty:
        acc = com_rec.oriented_acc.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)

    gyro = com_rec.rt_oriented_gyro.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)
    if gyro.empty:
        gyro = com_rec.oriented_gyro.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)

    gps = com_rec.gps.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)

    # extract tailgating scores
    tg = com_rec.tailgating.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)

    # extract distraction scores
    di = com_rec.dist_multilabel.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)

    # throw exception is any stream is missing
    missing_streams = []
    if acc.empty:
        missing_streams.append('oriented_acc')
    if gyro.empty:
        missing_streams.append('oriented_gyro')
    if gps.empty:
        missing_streams.append('gps')
    if tg.empty:
        missing_streams.append('tailgating')
    if di.empty:
        missing_streams.append('dist_multilabel')
    if len(missing_streams) > 0:
        raise MissingSensorStreamError(message=f"Sensor com_rec is missing {', '.join(missing_streams)} streams.",
                                       missing_streams=missing_streams)

    tg['score_tailgating'] = tg['score']
    tg.drop(columns=['score'], axis=1, inplace=True)

    # TODO: this currently works, since we interpolate all other sensors to acc.sensor_ns
    #  however, it is better to trim the sensor_ts to be the intersection of all sensors/videos

    # rename columns
    acc.rename(columns={c: f'acc_{c}' for c in ['x', 'y', 'z']}, inplace=True)
    gyro.rename(columns={c: f'gyr_{c}' for c in ['x', 'y', 'z']}, inplace=True)

    # remove system_ms column
    acc.drop(columns=['system_ms'], axis=1, inplace=True)
    gyro.drop(columns=['system_ms'], axis=1, inplace=True)
    gps.drop(columns=['system_ms'], axis=1, inplace=True)
    tg.drop(columns=['system_ms'], axis=1, inplace=True)
    di.drop(columns=['system_ms'], axis=1, inplace=True)

    # interpolate all sensor data and video scores to sensor_ns
    gyro_interp = interp_df(gyro, 'sensor_ns', acc.sensor_ns)
    gps_interp = interp_df(gps, 'sensor_ns', acc.sensor_ns)
    tg_interp = interp_df(tg, 'sensor_ns', acc.sensor_ns)
    di_interp = interp_df(di, 'sensor_ns', acc.sensor_ns)

    # merge different sensors based on sensor_ns
    # TODO: Check if this is proper way to do it
    join_col = 'sensor_ns'
    event_data = (acc
                  .merge(gyro_interp, on=join_col)
                  .merge(gps_interp, on=join_col)
                  .merge(tg_interp, on=join_col)
                  .merge(di_interp, on=join_col)
                  )

    return event_data
