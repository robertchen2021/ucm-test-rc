from typing import Dict
import numpy as np
from typing import Optional
from nauto_ds_utils.utils.data import find_closest_index


TO_DEG = 180 / np.pi


def compute_max_wz(ay_multiplier: float = 1.24,
                   r_min: float = 5.55,
                   v_slack: float = 5.0,
                   max_speed: float = 100.0,
                   step_size: float = 0.5,
                   G=9.80655
                   ) -> Dict[float, float]:
    """Speed Grip Model"""
    ay_max = ay_multiplier * G
    v = np.arange(-v_slack, max_speed, step_size)
    v_slack_1, v_slack_2 = abs(v - v_slack), abs(v + v_slack)

    wz_peak = np.sqrt(ay_max / r_min)

    wz_1 = ay_max / v_slack_1
    wz_2 = v_slack_2 / r_min
    wz_3 = np.ones(len(wz_1)) * wz_peak

    return dict(zip(v, np.min(np.c_[wz_1, wz_2, wz_3], axis=1) * TO_DEG))


def evaluate_speed_wz(speed: float,
                      wz: float,
                      speed_wz_mapping: Dict[float, float]) -> Optional[bool]:
    """Speed Look up and compare if the speed is vehicle dynamically 
    feasible. """
    try:
        v = list(speed_wz_mapping.keys())
        arg = find_closest_index(v, speed, min)
        return wz >= speed_wz_mapping[v[arg]]
    except TypeError:
        return None
