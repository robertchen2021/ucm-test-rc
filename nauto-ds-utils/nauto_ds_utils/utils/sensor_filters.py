# TODO:
# Add Moving Average Filter
# Finish Docstrings
# Finish type casting

from typing import Tuple
from numpy import ndarray
from scipy.signal import butter, lfilter


def butter_lowpass(cutOff, fs, order: int = 5) -> Tuple[ndarray]:
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    return butter(order, normalCutoff, btype='low', analog=False)


def butter_lowpass_filter(data, cutOff, fs, order: int = 4) -> ndarray:
    b, a = butter_lowpass(cutOff, fs, order=order)
    return lfilter(b, a, data)


def butter_highpass(cutOff, fs, order: int = 5) -> Tuple[ndarray]:
    normalCutoff = cutOff / (0.5 * fs)
    b, a = butter(order, normalCutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutOff, fs, order: int = 4) -> ndarray:
    b, a = butter_highpass(cutOff, fs, order=order)
    return lfilter(b, a, data)


def moving_average(data, N=20) -> ndarray:
    """
    Applies N-points moving average to the data
    Note: for the first i points, where i < N, take the cumsum(data[:i])/i
    Params: 
        data: 1D array, the data to be averaged
        N: int, the size of the window
    Returns:
        out: 1D array, after averaged
    """
    return np.hstack((np.cumsum(data[:N-1]) / np.arange(1, N),
                      np.convolve(data, np.ones((N,))/N, mode='valid'))
                     )