from typing import Optional

import numpy as np

from nauto_datasets.utils.numpy import NDArray


def delta_decompress(data: np.ndarray,
                     scale: Optional[float] = None) -> np.ndarray:
    """Performs delta decompression of an array of values.

    Args:
        data: `data` contains values which are (optionally) scaled differences
            between the original consecutive values
        scale: if provided then the resulting values sum are additionally
            multiplied by 1/scale
    Returns:
        delta decompressed data
    Raises:
         ValueError: when `data` is not empty and scale = 0
    """
    if len(data) == 0:
        return data

    summed = np.cumsum(data)
    if scale is None:
        return summed
    else:
        if scale == 0:
            raise ValueError('Cannot divide by scale = 0')
        return summed / scale


def to_utc_time(
        time_ns: NDArray[np.uint64],
        utc_boot_time_ns: np.uint64,
        utc_boot_time_offset_ns: np.int64,
        in_place: bool = False
) -> NDArray[np.uint64]:
    """Transforms recorded relative time in nano seconds to absolute utc time
    Args:
        time_ns: relative time in ns
        utc_boot_time_ns: absolute time of the sensors boot
        utc_boot_time_offset_ns: additional relative offset to consider

    Returns:
        utc_time_ns: time_ns expressed in absolute utc values
    """
    if in_place:
        new_time_ns = time_ns
    else:
        new_time_ns = np.copy(time_ns)
    new_time_ns += utc_boot_time_ns
    # casts to avoid conversions to float64
    if utc_boot_time_offset_ns > 0:
        new_time_ns += utc_boot_time_offset_ns.astype(np.uint64)
    else:
        new_time_ns -= np.abs(utc_boot_time_offset_ns).astype(np.uint64)

    return new_time_ns
