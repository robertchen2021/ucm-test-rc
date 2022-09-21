from itertools import chain
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import scipy as sp
from scipy.signal import butter

from .algos_imu import butter_lowpass, FindSevereBrakingEvents, DetectSpeedBump, EstimVehTrajRadius, \
    startle_braking_candidate
from .utils import get_consecutive_time_segments


def get_panic_brake_times_v2(time: np.ndarray,
                             accx: np.ndarray,
                             gpsspeed: np.ndarray,
                             startle_slope_th: float,
                             startle_a_max_th: float,
                             startle_a_mid_th: float,
                             startle_buffet_length: int) -> Tuple[List, List, List, List]:
    # braking events
    brake_pd = FindSevereBrakingEvents(0, 0, 0, 0, time, accx, gpsspeed)
    brake_times = []
    brake_severity = []
    if len(brake_pd) > 0:
        for item in brake_pd.to_dict(orient='records'):
            brake_times.append([item['StartTime_s'], item['EndTime_s']])
            brake_severity.append(item['BrakeSeverity'])

    # panic events
    [b, a] = butter_lowpass(5, 200, 4)
    accx_lpf = sp.signal.lfilter(b, a, accx)

    startled_segments = startle_braking_candidate(
        time, accx_lpf, startle_slope_th, startle_a_max_th, startle_a_mid_th, startle_buffet_length)

    panic_times = []
    panic_severity = []  # slope plays a "severity" role (severity == slope)
    for segment in startled_segments:
        panic_times.append([time[segment["start"]], time[segment["stop"]]])
        panic_severity.append(segment["slope"])

    return brake_times, brake_severity, panic_times, panic_severity


def get_speed_bump_times(time: np.ndarray,
                         accx: np.ndarray,
                         gpsspeed: np.ndarray) -> Tuple[List, List]:
    """
    Detect speed bumps and return speed bump times and severity
    :param time:
    :param accx:
    :param gpsspeed:
    :return:
    """
    sb_pd = DetectSpeedBump(time, accx, time, gpsspeed)

    if len(sb_pd) > 0:
        # extract and merge the potential overlapping indices
        all_indices = []
        for _, row in sb_pd.iterrows():
            all_indices += range(row['BumpStartIndices'], row['BumpEndIndices'] + 1)
        all_indices = sorted(np.unique(all_indices))
        # find consecutive segments and speed bump severity
        sb_times, sb_indices = get_consecutive_time_segments(time, all_indices)
        sb_severity = []
        for seg_indices in sb_indices:
            sev = max(sb_pd.BumpSeverities[(sb_pd.BumpStartIndices >= min(seg_indices)) &
                                           (sb_pd.BumpEndIndices <= max(seg_indices))])
            sb_severity.append(sev)
    else:
        sb_times = []
        sb_severity = []

    return sb_times, sb_severity


def get_turn_times(time: np.ndarray,
                   accy: np.ndarray,
                   gyrz: np.ndarray,
                   gpsspeed: np.ndarray,
                   turn_time_th: float = 0.5) -> Tuple[List, List]:
    """
    Detect turn and return turn times and max radii.
    :param time:
    :param accy:
    :param gyrz:
    :param gpsspeed:
    :param turn_time_th:
    :return:
    """
    # smooth signals
    [b, a] = butter_lowpass(10, 200, 4)
    lpf10_accy = sp.signal.filtfilt(b, a, accy)
    lpf10_gyrz = sp.signal.filtfilt(b, a, gyrz)

    turn_ang, r_imu, v_imu, r_imu_gps_speed = EstimVehTrajRadius(time, lpf10_accy, lpf10_gyrz, gpsspeed)
    turn_indices = np.where(abs(r_imu_gps_speed) > 0)[0]

    turn_times = []
    turn_radii = []
    if len(turn_indices) > 0:
        # find consecutive turn times
        all_times, all_seg_ind = get_consecutive_time_segments(time, turn_indices)

        # apply time threshold
        turn_times, turn_seg_ind = [], []
        for t_win, seg_ind in zip(all_times, all_seg_ind):
            if t_win[1] - t_win[0] > turn_time_th:
                turn_times.append(t_win)
                turn_seg_ind.append(seg_ind)

        if len(turn_times) > 0:
            # extract and merge the potential overlapping indices
            turn_radii = [max(abs(r_imu_gps_speed[i])) for i in turn_seg_ind]

    return turn_times, turn_radii


def get_tailgating_times(ed: pd.DataFrame,
                         score_th: float = -1,
                         duration_th: float = 0.5) -> Tuple[List, List]:
    """
    Detect tailgatings and minimum following distance
    :param ed:
    :param score_th:
    :param duration_th:
    :return:
    """
    cols = ['sensor_ns', 'front_box_index', 'distance_estimate', 'score_tailgating']
    # cols = ['event_id','sensor_ns','front_box_index','distance_estimate','score_tailgating']
    ed = ed[~ed.score_tailgating.isna()][cols].sort_values('sensor_ns').reset_index(drop=True)

    filters = ed.front_box_index > score_th
    if ed[filters].empty:
        return [], []

    ed['time_s'] = 1e-9 * (ed.sensor_ns - ed.sensor_ns.min())
    ed['time_idx'] = ed.index

    ed = (ed
          [filters]
          [['time_s', 'sensor_ns', 'front_box_index', 'distance_estimate', 'time_idx']]
          .sort_values('time_s')
          .reset_index(drop=True)
          )
    ed['seg_id'] = (ed.time_idx - ed.index)

    # identify tailgating segments
    segs = (ed
            .groupby('seg_id')
            .agg(min_ns=('sensor_ns', min),
                 max_ns=('sensor_ns', max),
                 duration=('time_s', lambda x: max(x) - min(x)),
                 min_distance=('distance_estimate', min))
            .reset_index()
            )

    time_segs = []
    min_dist = []
    for _, row in segs[segs.duration > duration_th].iterrows():
        time_segs.append([int(row['min_ns']), int(row['max_ns'])])
        min_dist.append(float(row['min_distance']))
    return time_segs, min_dist


def get_distraction_times(ed: pd.DataFrame,
                          score_th: float = 0.5,
                          duration_th: float = 1.0) -> Tuple[List, List]:
    """
    Detect distractions and the maximum distraction score
    :param ed:
    :param score_th:
    :param duration_th:
    :return:
    """
    look_away_cols = ['score_looking_down', 'score_looking_left', 'score_looking_right', 'score_looking_up']
    cols = ['sensor_ns'] + look_away_cols
    # cols = ['event_id','sensor_ns']+look_away_cols
    ed = ed[~ed.score_looking_down.isna()][cols].sort_values('sensor_ns').reset_index(drop=True)

    # new score: looking away
    ed['score_looking_away'] = ed[look_away_cols].max(axis=1)
    filters = ed.score_looking_away > score_th
    if ed[filters].empty:
        return [], []

    ed['time_s'] = 1e-9 * (ed.sensor_ns - ed.sensor_ns.min())
    ed['time_idx'] = ed.index

    ed = (ed
          [filters]
          [['time_s', 'sensor_ns', 'score_looking_away', 'time_idx']]
          .sort_values('time_s')
          .reset_index(drop=True)
          )
    ed['seg_id'] = (ed.time_idx - ed.index)

    # identify distraction segments
    segs = (ed
            .groupby('seg_id')
            .agg(min_ns=('sensor_ns', min),
                 max_ns=('sensor_ns', max),
                 duration=('time_s', lambda x: max(x) - min(x)),
                 max_score=('score_looking_away', max))
            .reset_index()
            )

    time_segs = []
    max_score = []
    for _, row in segs[segs.duration > duration_th].iterrows():
        time_segs.append([int(row['min_ns']), int(row['max_ns'])])
        max_score.append(float(row['max_score']))
    return time_segs, max_score


def get_holding_object_times(ed: pd.DataFrame,
                             score_th: float = 0.5,
                             duration_th: float = 1.0) -> Tuple[List, List]:
    """
    Detect holding objects and the scores
    :param ed:
    :param score_th:
    :param duration_th:
    :return:
    """
    holding_cols = ['score_cell_phone', 'score_holding_object']
    cols = ['sensor_ns'] + holding_cols
    # cols = ['event_id','sensor_ns']+holding_cols
    ed = ed[~ed.score_cell_phone.isna()][cols].sort_values('sensor_ns').reset_index(drop=True)

    # new score: holding object
    ed['score_holding_object'] = ed[holding_cols].max(axis=1)
    filters = ed.score_holding_object > score_th
    if ed[filters].empty:
        return [], []

    ed['time_s'] = 1e-9 * (ed.sensor_ns - ed.sensor_ns.min())
    ed['time_idx'] = ed.index

    ed = (ed
          [filters]
          [['time_s', 'sensor_ns', 'score_holding_object', 'time_idx']]
          .sort_values('time_s')
          .reset_index(drop=True)
          )
    ed['seg_id'] = (ed.time_idx - ed.index)

    # identify distraction segments
    segs = (ed
            .groupby('seg_id')
            .agg(min_ns=('sensor_ns', min),
                 max_ns=('sensor_ns', max),
                 duration=('time_s', lambda x: max(x) - min(x)),
                 max_score=('score_holding_object', max))
            .reset_index()
            )

    time_segs = []
    max_score = []
    for _, row in segs[segs.duration > duration_th].iterrows():
        time_segs.append([int(row['min_ns']), int(row['max_ns'])])
        max_score.append(float(row['max_score']))
    return time_segs, max_score


def get_no_face_times(ed: pd.DataFrame,
                      score_th: float = 0.5,
                      duration_th: float = 1.0) -> Tuple[List, List]:
    """
    Detect no face times and the scores
    :param ed:
    :param score_th:
    :param duration_th:
    :return:
    """
    holding_cols = ['score_no_face']
    cols = ['sensor_ns'] + holding_cols
    # cols = ['event_id', 'sensor_ns'] + holding_cols
    ed = ed[~ed.score_no_face.isna()][cols].sort_values('sensor_ns').reset_index(drop=True)

    # new score: holding object
    filters = ed.score_no_face > score_th
    if ed[filters].empty:
        return [], []

    ed['time_s'] = 1e-9 * (ed.sensor_ns - ed.sensor_ns.min())
    ed['time_idx'] = ed.index

    ed = ed[filters][['time_s', 'sensor_ns', 'score_no_face', 'time_idx']].sort_values('time_s').reset_index(drop=True)
    ed['seg_id'] = (ed.time_idx - ed.index)

    # identify distraction segments
    segs = (ed
            .groupby('seg_id')
            .agg(min_ns=('sensor_ns', min),
                 max_ns=('sensor_ns', max),
                 duration=('time_s', lambda x: max(x) - min(x)),
                 max_score=('score_no_face', max))
            .reset_index()
            )

    time_segs = []
    max_score = []
    for _, row in segs[segs.duration > duration_th].iterrows():
        time_segs.append([int(row['min_ns']), int(row['max_ns'])])
        max_score.append(float(row['max_score']))
    return time_segs, max_score


def get_coachable_times(tailgating_times_ns: List[List],
                        tailgating_min_distance: List,
                        distraction_times_ns: List[List],
                        distraction_max_score: List,
                        startle_times_ns: List[List],
                        startle_severity: List,
                        time_delta_ns: int = 5e9) -> Tuple[List, List]:
    """
    Determine if the event is coachable, based on the sub-filters
    :param tailgating_times_ns:
    :param tailgating_min_distance:
    :param distraction_times_ns:
    :param distraction_max_score:
    :param startle_times_ns:
    :param startle_severity:
    :param time_delta_ns:
    :return:
    """
    coachable_times_ns = []
    coachable_severity = []

    # if either of the three component is empty, not a coachable event
    if (len(tailgating_times_ns) == 0) or (len(distraction_times_ns) == 0) or (len(startle_times_ns) == 0):
        return coachable_times_ns, coachable_severity

    # 2021/01/12 - disabled that condition, because no data to check it right now.
    # if more than 1 startle, typically a device error and not a coachable event
    # if len(startle_times_ns) > 1:
    #     return coachable_times_ns, coachable_severity

    flat_tg_ns = list(chain.from_iterable(tailgating_times_ns))
    flat_di_ns = list(chain.from_iterable(distraction_times_ns))
    flat_st_ns = list(chain.from_iterable(startle_times_ns))

    # must have distraction and tailgating ahead of startle
    if (min(flat_st_ns) <= min(flat_di_ns)) or (min(flat_st_ns) <= min(flat_tg_ns)):
        return coachable_times_ns, coachable_severity

    # from tailgating, distraction, startle, determine the coachable_times_ns and coachable_severity
    # for now, just use some random numbers so we complete the functional test:
    # TODO: MUST update the logic that determines the coachable_times_ns and coachable_severity
    coachable_times_ns.append([min(min(flat_tg_ns), min(flat_di_ns), min(flat_st_ns)),
                               max(max(flat_tg_ns), max(flat_di_ns), max(flat_st_ns))])

    coachable_severity.append(min(tailgating_min_distance) + max(distraction_max_score) + max(startle_severity))

    return coachable_times_ns, coachable_severity
