"""
This file contains some util tools that are used on the processing functions
com_rec.rt_* is used for config_version >= 3.4
com_rec.* is used for config_version < 3.4
TODO: remove com_rec.* checking after all devices get updated to 3.4
"""

import numpy as np
import pandas as pd

from nauto_datasets.core.sensors import CombinedRecording
from nauto_zoo import MissingSensorStreamError


def interp_df(df_in: pd.DataFrame, x_col_name: str, x_new_vals: np.ndarray) -> pd.DataFrame:
    if x_col_name not in df_in.columns:
        raise ValueError(f'x_col_name: "{x_col_name}" not in the target dataframe')

    df_out = pd.DataFrame()
    df_out[x_col_name] = x_new_vals

    for col in df_in.columns:
        if col != x_col_name:
            df_out[col] = np.interp(x_new_vals, df_in[x_col_name], df_in[col])

    return df_out


def com_rec_to_df(com_rec: 'CombinedRecording') -> pd.DataFrame:
    """
    Parse combined records for sensors and video scores, and store as pandas dataframe
    :param com_rec:
    :return: dataframe, sorted based on "sensor_ns"
    """

    sensor_ns = com_rec.rt_oriented_acc.stream.sensor_ns
    if len(sensor_ns) == 0:
        sensor_ns = com_rec.oriented_acc.stream.sensor_ns

    sensor_ns = np.sort(sensor_ns)

    tg = com_rec.tailgating.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)
    di = com_rec.dist_multilabel.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)

    missing_streams = []
    if len(sensor_ns) == 0:
        missing_streams.append('acc.sensor_ns')
    if tg.empty:
        missing_streams.append('tailgating')
    if di.empty:
        missing_streams.append('dist_multilabel')
    if len(missing_streams) > 0:
        raise MissingSensorStreamError(message=f"Sensor com_rec is missing {', '.join(missing_streams)} streams.",
                                       missing_streams=missing_streams)

    tg['score_tailgating'] = tg['score']
    tg.drop(columns=['score'], axis=1, inplace=True)

    tg.drop(columns=['system_ms'], axis=1, inplace=True)
    di.drop(columns=['system_ms'], axis=1, inplace=True)

    tg_interp = interp_df(tg, 'sensor_ns', sensor_ns)
    di_interp = interp_df(di, 'sensor_ns', sensor_ns)

    # merge different sensors based on sensor_ns
    # TODO: Check if this is proper way to do it
    join_col = 'sensor_ns'
    event_data = tg_interp.merge(di_interp, on=join_col)

    return event_data
