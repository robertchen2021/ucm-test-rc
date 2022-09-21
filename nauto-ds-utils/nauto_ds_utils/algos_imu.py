from typing import List, Dict

import numpy as np


def startle_braking_candidate(time_sec: np.ndarray,
                              accx_lpf: np.ndarray,
                              slope_min: float = 3.25,
                              a_max_th: float = -6.25,
                              a_mid_th: float = -3.,
                              buffet_length: int = 200, ) -> List[Dict]:
    """
    Here we try to detect just the "startle braking" candidate because the real "startle braking" can be determined
     only based on the video.
    """
    n = len(accx_lpf)
    i1 = -1
    i2 = buffet_length - 2
    startled_segments = []
    while i2 < n - 1:
        i1 += 1
        i2 += 1
        if accx_lpf[i2] > a_max_th:
            continue
        dt = time_sec[i2] - time_sec[i1]
        da = accx_lpf[i1] - accx_lpf[i2]
        slope = da / dt
        if slope < slope_min:
            continue
        startled_segments.append(
            {
                "start": i1,
                "stop": i2,
                "dt": dt,
                "slope": slope,
            }
        )
        while i2 < n and accx_lpf[i2] < a_mid_th:
            i1 += 1
            i2 += 1

    return startled_segments
