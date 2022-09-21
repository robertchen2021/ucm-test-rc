# python 3.6

from typing import Any, Dict, List, Tuple
# from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import scipy as sp
import pandas as pd
from scipy.signal import butter
from collections import OrderedDict
from itertools import chain
from algos_imu import butter_lowpass, FindPanicBrakingEvents, DetectStartle, DetectSpeedBump, EstimVehTrajRadius
from utils import get_consecutive_time_segments, sec_to_sensor_ns, com_rec_to_df
from nauto_datasets.core.sensors import CombinedRecording


# The main idea of this model is to detect "tailgating+distraction+start-brake"
# The core functions are leveraged from Markus's qubole notebook: 
# https://us.qubole.com/notebooks#home?id=36986
#
# Initially, we will attempt to create a model that runs end-to-end,  similar to how the startle-filter works:
#   It takes as input the on-device braking-hard events, 
#   then processes the sensor data to detect tailgating, distraction, start-brake,
#   then provides an overall evaluation if is_tg==True & is_dist==True & is_sb==True.
#   It returns the labels, severities and time_windows of the overall evaluation, along with the components
#   (tailgating, distraction, startle-brake) 

# We will later attempt to modularize by creating individual cloud models of the subtask,
# then chain the cloud models in sequence.

##############################
# Processing functions
##############################
def get_panic_brakes(time: np.ndarray,
                     accx: np.ndarray,
                     gpsspeed: np.ndarray) -> Tuple[List, List, List, List]:
    """
    Call panic brake detection and format outputs
    :param time:
    :param accx:
    :param gpsspeed:
    :return:
    """
    # filtered signals
    [b, a] = butter_lowpass(10, 200, 4)
    lpf10_accx = sp.signal.filtfilt(b, a, accx)
    [b, a] = butter_lowpass(20, 200, 4)
    lpf20_accx = sp.signal.filtfilt(b, a, accx)

    # braking events
    brake_pd = FindPanicBrakingEvents(0, 0, 0, 0, time, accx, gpsspeed)
    brake_times = []
    brake_severity = []
    if len(brake_pd) > 0:
        for item in brake_pd.to_dict(orient='records'):
            brake_times.append([item['StartTime_s'], item['EndTime_s']])
            brake_severity.append(item['BrakeSeverity'])

    # panic events
    PANIC, BUMP, panicTrue, panicTimes, panicIndices, panicSeverity, Num_ExpectedBumpStartTimes, \
    ExpectedBumpStartTimes, ExpectedBumpEndTimes, BumpPanicTimes, \
    TRUE_PANIC_FOUND, TruePanicIndices = DetectStartle(time, lpf10_accx, lpf20_accx)

    if TRUE_PANIC_FOUND:
        panic_times, panic_indices = get_consecutive_time_segments(time, TruePanicIndices[TruePanicIndices > 0])
        # extract panic severity for each panic-segment
        panic_severity = []
        for segment_indices in panic_indices:
            idx = (TruePanicIndices >= segment_indices[0]) & (TruePanicIndices <= segment_indices[-1])
            panic_severity.append(min(panicSeverity[idx]))
    else:
        panic_times = []
        panic_severity = []

    return brake_times, brake_severity, panic_times, panic_severity


def get_startle(time: np.ndarray,
                accx: np.ndarray) -> Tuple[List, List]:
    """
    Call startle detection and format output
    :param time:
    :param accx:
    :return:
    """
    # filtered signals
    [b, a] = butter_lowpass(10, 200, 4)
    lpf10_accx = sp.signal.filtfilt(b, a, accx)
    [b, a] = butter_lowpass(20, 200, 4)
    lpf20_accx = sp.signal.filtfilt(b, a, accx)

    # panic events
    PANIC, BUMP, panicTrue, panicTimes, panicIndices, panicSeverity, Num_ExpectedBumpStartTimes, \
    ExpectedBumpStartTimes, ExpectedBumpEndTimes, BumpPanicTimes, \
    TRUE_PANIC_FOUND, TruePanicIndices = DetectStartle(time, lpf10_accx, lpf20_accx)

    if TRUE_PANIC_FOUND:
        panic_times, panic_indices = get_consecutive_time_segments(time, TruePanicIndices[TruePanicIndices > 0])
        # extract panic severity for each panic-segment
        panic_severity = []
        for segment_indices in panic_indices:
            idx = (TruePanicIndices >= segment_indices[0]) & (TruePanicIndices <= segment_indices[-1])
            panic_severity.append(min(panicSeverity[idx]))
    else:
        panic_times = []
        panic_severity = []

    return panic_times, panic_severity


def get_speed_bumps(time: np.ndarray,
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


def get_turns(time: np.ndarray,
              accy: np.ndarray,
              gyrz: np.ndarray,
              gpsspeed: np.ndarray,
              time_threshold: float = 0.5) -> Tuple[List, List]:
    """
    Detect turn and return turn times and max radii.
    :param time:
    :param accy:
    :param gyrz:
    :param gpsspeed:
    :param time_threshold:
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
            if t_win[1] - t_win[0] > time_threshold:
                turn_times.append(t_win)
                turn_seg_ind.append(seg_ind)

        if len(turn_times) > 0:
            # extract and merge the potential overlapping indices
            turn_radii = [max(abs(r_imu_gps_speed[i])) for i in turn_seg_ind]

    return turn_times, turn_radii


def get_tailgating_times(ed: pd.DataFrame,
                         time_thresh: float = 0.5) -> Tuple[List, List]:
    
    cols = ['sensor_ns', 'front_box_index', 'distance_estimate', 'score_tailgating']
    # cols = ['event_id','sensor_ns','front_box_index','distance_estimate','score_tailgating']
    ed = ed[~ed.score_tailgating.isna()][cols].sort_values('sensor_ns').reset_index(drop=True)
    
    filters = ed.front_box_index > -1
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
    for _, row in segs[segs.duration > time_thresh].iterrows():
        time_segs.append([int(row['min_ns']), int(row['max_ns'])])
        min_dist.append(float(row['min_distance']))
    return time_segs, min_dist

    
def get_distraction_times(ed: pd.DataFrame,
                          time_thresh: float = 1.0,
                          distract_thresh: float = 0.5) -> Tuple[List, List]:
   
    look_away_cols = ['score_looking_down', 'score_looking_left', 'score_looking_right', 'score_looking_up']
    cols = ['sensor_ns'] + look_away_cols
    # cols = ['event_id','sensor_ns']+look_away_cols
    ed = ed[~ed.score_looking_down.isna()][cols].sort_values('sensor_ns').reset_index(drop=True)

    # new score: looking away
    ed['score_looking_away'] = ed[look_away_cols].max(axis=1)
    filters = ed.score_looking_away > distract_thresh
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
    for _, row in segs[segs.duration > time_thresh].iterrows():
        time_segs.append([int(row['min_ns']), int(row['max_ns'])])
        max_score.append(float(row['max_score']))
    return time_segs, max_score
    
    
def get_holding_object_times(ed: pd.DataFrame,
                             time_thresh: float = 1.0,
                             distract_thresh: float = 0.5) -> Tuple[List, List]:
   
    holding_cols = ['score_cell_phone', 'score_holding_object']
    cols = ['sensor_ns'] + holding_cols
    # cols = ['event_id','sensor_ns']+holding_cols
    ed = ed[~ed.score_cell_phone.isna()][cols].sort_values('sensor_ns').reset_index(drop=True)

    # new score: holding object
    ed['score_holding_object'] = ed[holding_cols].max(axis=1)
    filters = ed.score_holding_object > distract_thresh
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
    for _, row in segs[segs.duration > time_thresh].iterrows():
        time_segs.append([int(row['min_ns']), int(row['max_ns'])])
        max_score.append(float(row['max_score']))
    return time_segs, max_score


def get_no_face_times(ed: pd.DataFrame,
                      time_thresh: float = 1.0,
                      score_thresh: float = 0.5) -> Tuple[List, List]:

    holding_cols = ['score_no_face']
    cols = ['sensor_ns'] + holding_cols
    # cols = ['event_id', 'sensor_ns'] + holding_cols
    ed = ed[~ed.score_no_face.isna()][cols].sort_values('sensor_ns').reset_index(drop=True)

    # new score: holding object
    filters = ed.score_no_face > score_thresh
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
    for _, row in segs[segs.duration > time_thresh].iterrows():
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

    flat_tg_ns = list(chain.from_iterable(tailgating_times_ns))
    flat_di_ns = list(chain.from_iterable(distraction_times_ns))
    flat_st_ns = list(chain.from_iterable(startle_times_ns))

    # from tailgating, distraction, startle, determine the coachable_times_ns and coachable_severity
    # for now, just use some random numbers so we complete the functional test:
    # TODO: MUST update the logic that determines the coachable_times_ns and coachable_severity
    coachable_times_ns.append([min(min(flat_tg_ns), min(flat_di_ns), min(flat_st_ns)),
                               max(max(flat_tg_ns), max(flat_di_ns), max(flat_st_ns))])

    coachable_severity = min(tailgating_min_distance) + max(distraction_max_score) + max(startle_severity)

    return coachable_times_ns, coachable_severity


# def process_video_labels(event_vs: pd.DataFrame) -> Dict[str, Any]:
#     tailgating_times_ns, tailgating_min_distance = get_tailgating_times(event_vs, time_thresh=0.5)
#     distraction_times_ns, distraction_max_score = get_distraction_times(event_vs)
#     holding_object_times_ns, holding_object_max_score = get_holding_object_times(event_vs)
#     no_face_times_ns, no_face_max_score = get_no_face_times(event_vs)
#
#     rec = OrderedDict({
#         'device_id': event_vs.device_id.iloc[0],
#         'event_id': event_vs.event_id.iloc[0],
#         'tailgating_times_ns': tailgating_times_ns,
#         'tailgating_min_dist': tailgating_min_distance,
#         'looking_away_times_ns': distraction_times_ns,
#         'looking_away_max_score': distraction_max_score,
#         'holding_object_times_ns': holding_object_times_ns,
#         'holding_object_max_score': holding_object_max_score,
#         'no_face_times_ns': no_face_times_ns,
#         'no_face_max_score': no_face_max_score,
#     })
#     return rec
   

# def process_one_event(imu: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     process the sensor data from one event
#     :param imu: dictionary with imu sensor data
#     :return: Dictionary with startle times, severity and sensor offset
#     """
#     # define output
#     rec = OrderedDict({
#             'startle_times_ns': [],
#             'startle_severity': [],
#             'sensor_offset_ns': None
#             })
#
#     if not imu:
#         return rec
#
#     # extract sensor data
#     sensor_offset_ns = int(imu['sensor_ns'].min())
#     time_s = (imu['sensor_ns'] - sensor_offset_ns) * 1e-9
#     acc_x = imu['acc_x']
#     if time_s.shape != acc_x.shape:
#         raise(ValueError('Signals data does not match'))
#
#     # sort by time
#     sort_idx = np.argsort(time_s)
#     time_s = time_s[sort_idx]
#     acc_x = acc_x[sort_idx]
#     # get startle brakes
#     startle_times, startle_severity = get_startle(time_s, acc_x)
#
#     # convert to sensor time ns
#     startle_times_ns = sec_to_sensor_ns(startle_times, sensor_offset_ns)
#
#     rec['startle_times_ns'] = startle_times_ns
#     rec['startle_severity'] = startle_severity
#     rec['sensor_offset_ns'] = sensor_offset_ns
#
#     return rec


# def process_one_event_new(event_data: pd.DataFrame) -> Dict[str, Any]:
#     """
#     process the sensor data from one event
#     :param event_data:
#     :return:
#     """
#     event_data.sort_values('sensor_ns', inplace=True)
#     event_data.reset_index(inplace=True, drop=True)
#
#     sensor_offset_ns = event_data.sensor_ns.min()
#     event_data['time_s'] = (event_data.sensor_ns - sensor_offset_ns) * 1e-9
#     device_id = event_data.device_id.unique().tolist()[0]
#     event_id = event_data.event_id.unique().tolist()[0]
#     time, accx, accy, gyrz, gpsspeed = \
#         [event_data[col].values for col in ['time_s', 'acc_x', 'acc_y', 'gyr_z', 'speed']]
#     # print([len(x) for x in [time, accx, accy, gyrz, gpsspeed]])
#
#     # brakes and startle brakes
#     brake_times, brake_severity, startle_times, startle_severity = get_panic_brakes(time, accx, gpsspeed)
#     # speed bump
#     sb_times, sb_severity = get_speed_bumps(time, accx, gpsspeed)
#     # turns
#     turn_times, turn_radii = get_turns(time, accy, gyrz, gpsspeed)
#
#     # convert to sensor time ns
#     brake_times_ns, startle_times_ns, sb_times_ns, turn_times_ns = \
#         [sec_to_sensor_ns(t, sensor_offset_ns) for t in [brake_times, startle_times, sb_times, turn_times]]
#
#     rec = OrderedDict({
#         'device_id': device_id,
#         'event_id': event_id,
#         'brake_times_ns': brake_times_ns,
#         'brake_severity': brake_severity,
#         'startle_times_ns': startle_times_ns,
#         'startle_severity': startle_severity,
#         'speed_bump_times_ns': sb_times_ns,
#         'speed_bump_severity': sb_severity,
#         'turn_times_ns': turn_times_ns,
#         'turn_radii': turn_radii,
#         'sensor_offset_ns': sensor_offset_ns
#     })
#     return rec


def process_one_event_merged(com_rec: 'CombinedRecording') -> Dict[str, Any]:
    """
    Process the sensor data and video_scores from one event
    :param com_rec: combined recording of an event
    :return: rec, an ordered dictionary which stores the detection outcome
    """

    # Define the output structure
    rec = OrderedDict({
        'device_id': '',
        'event_id': '',
        'sensor_offset_ns': None,
        'coachable_times_ns': [],
        'coachable_severity': [],
        'brake_times_ns': [],
        'brake_severity': [],
        'startle_times_ns': [],
        'startle_severity': [],
        'speed_bump_times_ns': [],
        'speed_bump_severity': [],
        'turn_times_ns': [],
        'turn_radii': [],
        'tailgating_times_ns': [],
        'tailgating_min_dist': [],
        'looking_away_times_ns': [],
        'looking_away_max_score': [],
        'holding_object_times_ns': [],
        'holding_object_max_score': [],
        'no_face_times_ns': [],
        'no_face_max_score': [],
    })
    if not com_rec:
        return rec

    # Extract sensor data and video_scores from the com_rec
    event_data = com_rec_to_df(com_rec)
    # event_data['device_id'] = device_id
    # event_data['event_id'] = event_id

    # Get the sensor_offset_ns
    sensor_offset_ns = int(event_data.sensor_ns.min())

    # Process for imu-based algos
    event_data.sort_values('sensor_ns', inplace=True)
    event_data.reset_index(inplace=True, drop=True)

    event_data['time_s'] = (event_data.sensor_ns - sensor_offset_ns) * 1e-9
    # device_id = event_data.device_id.unique().tolist()[0]
    # event_id = event_data.event_id.unique().tolist()[0]
    time, accx, accy, gyrz, gpsspeed = \
        [event_data[col].values for col in ['time_s', 'acc_x', 'acc_y', 'gyr_z', 'speed']]
    # print([len(x) for x in [time, accx, accy, gyrz, gpsspeed]])

    # brakes and startle brakes
    brake_times, brake_severity, startle_times, startle_severity = get_panic_brakes(time, accx, gpsspeed)
    # speed bump
    sb_times, sb_severity = get_speed_bumps(time, accx, gpsspeed)
    # turns
    turn_times, turn_radii = get_turns(time, accy, gyrz, gpsspeed)

    # convert to sensor time ns
    brake_times_ns, startle_times_ns, sb_times_ns, turn_times_ns = \
        [sec_to_sensor_ns(t, sensor_offset_ns) for t in [brake_times, startle_times, sb_times, turn_times]]

    # Process for video-based algos
    tailgating_times_ns, tailgating_min_distance = get_tailgating_times(event_data, time_thresh=0.5)
    distraction_times_ns, distraction_max_score = get_distraction_times(event_data)
    holding_object_times_ns, holding_object_max_score = get_holding_object_times(event_data)
    no_face_times_ns, no_face_max_score = get_no_face_times(event_data)

    # Make judgement on "coachable"
    # for v0 model, we only use tailgating, distraction, startle-braking
    coachable_times_ns, coachable_severity = get_coachable_times(tailgating_times_ns, tailgating_min_distance,
                                                                 distraction_times_ns, distraction_max_score,
                                                                 startle_times_ns, startle_severity,
                                                                 time_delta_ns=int(5e9))

    # Package into output
    rec['sensor_offset_ns'] = sensor_offset_ns
    rec['coachable_times_ns'] = coachable_times_ns
    rec['coachable_severity'] = coachable_severity
    rec['brake_times_ns'] = brake_times_ns
    rec['brake_severity'] = brake_severity
    rec['startle_times_ns'] = startle_times_ns
    rec['startle_severity'] = startle_severity
    rec['speed_bump_times_ns'] = sb_times_ns
    rec['speed_bump_severity'] = sb_severity
    rec['turn_times_ns'] = turn_times_ns
    rec['turn_radii'] = turn_radii
    rec['tailgating_times_ns'] = tailgating_times_ns
    rec['tailgating_min_dist'] = tailgating_min_distance
    rec['looking_away_times_ns'] = distraction_times_ns
    rec['looking_away_max_score'] = distraction_max_score
    rec['holding_object_times_ns'] = holding_object_times_ns
    rec['holding_object_max_score'] = holding_object_max_score
    rec['no_face_times_ns'] = no_face_times_ns
    rec['no_face_max_score'] = no_face_max_score

    return rec
