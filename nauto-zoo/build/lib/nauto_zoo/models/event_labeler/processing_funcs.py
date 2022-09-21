from typing import List, Tuple

import pandas as pd


def get_time_intervals(ed: pd.DataFrame,
                       score_th: float,
                       subject_cols: List[str],
                       duration_th: float) -> Tuple[List, List]:
    """
    Detects suspect time intervals and the maximum scores based on subject_cols and thresholds.
    """

    cols = ['sensor_ns'] + subject_cols
    # TODO: It seems sort_values is unnecessary.
    ed = ed[~ed.score_looking_down.isna()][cols].sort_values('sensor_ns').reset_index(drop=True)

    ed['score'] = ed[subject_cols].max(axis=1)
    filters = ed.score > score_th
    if ed[filters].empty:
        return [], []

    ed['time_s'] = 1e-9 * (ed.sensor_ns - ed.sensor_ns.min())
    ed['time_idx'] = ed.index

    ed = (ed
          [filters]
          [['time_s', 'sensor_ns', 'score', 'time_idx']]
          .reset_index(drop=True)
          )
    ed['seg_id'] = (ed.time_idx - ed.index)

    segs = (ed
            .groupby('seg_id')
            .agg(min_ns=('sensor_ns', min),
                 max_ns=('sensor_ns', max),
                 duration=('time_s', lambda x: max(x) - min(x)),
                 max_score=('score', max))
            .reset_index()
            )

    time_segs = []
    max_score = []
    for _, row in segs[segs.duration > duration_th].iterrows():
        time_segs.append([int(row['min_ns']), int(row['max_ns'])])
        max_score.append(float(row['max_score']))
    return time_segs, max_score


def get_tailgating_time_intervals(ed: pd.DataFrame,
                                  subject_cols: List[str],
                                  score_th: float = -1,
                                  duration_th: float = 0.5) -> Tuple[List, List]:
    """
    Detect tailgating and minimum following distance.
    The tailgating algo is slightly different from the other, so we keep separate realization of it.
    """

    cols = ['sensor_ns'] + subject_cols
    # TODO: it seems sort_values is unnecessary
    ed = ed[~ed.score_tailgating.isna()][cols].sort_values('sensor_ns').reset_index(drop=True)

    filters = ed.front_box_index > score_th
    if ed[filters].empty:
        return [], []

    ed['time_s'] = 1e-9 * (ed.sensor_ns - ed.sensor_ns.min())
    ed['time_idx'] = ed.index

    ed = (ed
          [filters]
          [['time_s', 'sensor_ns', 'front_box_index', 'distance_estimate', 'time_idx']]
          .reset_index(drop=True)
          )
    ed['seg_id'] = (ed.time_idx - ed.index)

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
