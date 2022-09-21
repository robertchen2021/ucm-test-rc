from scipy.signal import butter, lfilter, freqz, filtfilt, freqs, argrelextrema
from nauto_ds_utils.utils.sensor_filters import butter_lowpass, butter_highpass
import numpy as np

from typing import List, Dict, Tuple, Any, Optional

from nauto_datasets.utils.boto import BotoS3Client
from nauto_datasets.core.sensors import CombinedRecording
from nauto_ds_utils.utils.sensor import get_combined_recording
from nauto_ds_utils.preprocessing.sensor import RawIMUPreprocess
from nauto_ds_utils.utils.data import find_closest_index


class LooseDetector():
    '''
    This is Erwin's Loose Device Detection in standalone form. To use this:

    LooseDetector() will give you a preprocessed acc/gyro and if you want the
    judgements of the 

    This is for Serial Loose Device Detection. Process each event sequentially.
    This class is a collection of all the loose device detectors. It includes
    two parts: preprocessing IMU and loose device detectors.
    Parameters:
    :param protos: Sensor Protobuf Files
    :param acc_field: ACCEPTED field names: 'acc', 'rt_acc'
    :param gyro_field: ACCEPTED field names: 'gyro' 'rt_gyro'
    How to Read:
    1.  __init__
    2. _preprocess_imu(): this function will grab all the attributes from the
    class and preprocess the IMU for Loose detectors. Currently, we treat
    ACC and GYRO to be separate n x 4 matricies. Ideally, we can treat it
    as n x 6 matrix. (Keeping sensor_ns timing constant and interpolate the
    missing data points.)
    3. `serial_run()` will evaluate the event's sensor protos. It will return
    a dictionary with boolean results.
    Objective: We want to use this Loose Detector to figure out if an event
    is physically feasible for the vehicle. With that said, each individual
    detector has its own positive/negative classification. In the case where
    the event is false positively triggered, it's likely the device isn't
    observing a hard acceleration, braking-hard, or corner-left/right-hard.
    In essence, this is a v0 of a spam filter for our event feed or our
    event feed.
    [TODOs]
    1. Maintain only numpy arrays/vectors/matrices
    2. [BIG IMPACT] vectorize all matrix and vector operations
    3. Make sure to limit the number of copying (pass by object reference)
    4. Need to Consider Vehicle Type. Affected Classifier: `yaw_jitter`.
    5. Add Crashnet into list of detectors. the acc and gyro attributes are in
    the right forma
    6. Split off the preprocessing into another class and combined them into
    one gigantic n x 6 matrix. (currently it's two n x 4 matrix)
    7. Need to consider logic for oriented stream
    8. Eliminate the need to use cos and sin to evaluate for angles. But in
    terms of performance, it will not matter -- ~10 micrseonds
    '''

    def __init__(self,
                 protos: List[str],
                 acc_field: str = 'acc',
                 gyro_field: str = 'gyro'):

        self.sensor_protos = protos
        self.components = ['sensor_ns', 'x', 'y', 'z']
        self.acc_field = acc_field
        self.gyro_field = gyro_field
        self._preprocess_imu()

    def _preprocess_imu(self) -> CombinedRecording:
        '''
        '''
        s3 = BotoS3Client()
        com_rec = get_combined_recording(self.sensor_protos).to_utc_time()
        self.acc = self._sort_imu(com_rec, self.acc_field)
        self.gyro = self._sort_imu(com_rec, self.gyro_field)
        self.get_angle()
        m = self.get_rot_mat().T
        self.acc = self.acc.dot(m)
        self.gyro = self.gyro.dot(m)

    def get_rot_mat(self) -> np.ndarray:
        '''
        4x4 rotation matrix -- using euler angles to rotate the x, z channels.
        sensor_ns, and y are invariant in this matrix.
        '''
        m = np.identity(4)
        m[1][1], m[1][3] = np.cos(self.phi_xz), -np.sin(self.phi_xz)
        m[3][1], m[3][3] = -m[1][3], m[1][1]
        return m

    def _sort_imu(self,
                  com_rec: CombinedRecording,
                  field: str) -> np.ndarray:
        '''
        Currently, when we combine multiple deserialized sensor files, there
        will be overlap. Hence, some recordings will be out of sync/overlapping.
        We will need to
        '''
        tmp = np.c_[[getattr(getattr(com_rec, field).stream, x)
                     for x in self.components]]
        return tmp.T[tmp[0, :].argsort()]

    def get_angle(self) -> None:
        '''
        Renamed `Calc_DevAng`. This creates the angle (degrees) and phi_xz
        (radians betwen x and z axes.)
        [TODO]: If we can figure out how to re-orient the matrix without computing
        sin(self.phi_xz) ; that would be ideal since cos(self.phi_xz) is the
        same as `arg` in this case. If that can be achieved, we can remove
        `check_pitch_angle` and this function as well.
        '''

        g_xz = np.sqrt(np.median(np.square(self.acc[:, 1]) +
                                 np.square(self.acc[:, 3])))
        arg = np.median(self.acc[:, 3] / g_xz)

        if (arg > 1) | (arg < -1):
            phixz = np.arccos(np.sign(arg))
        else:
            phixz = np.arccos(arg)

        self.angle = phixz * 180 / np.pi
        self.phi_xz = phixz

    def check_pitch_angle(self) -> bool:
        '''Replacement for CheckIf_LooseOri
        [DEBUG]:
        return `angle`
        '''

        return False if 115.0 <= self.angle <= 160.0 else True

    @staticmethod
    def sudden_angle_change(signal: np.ndarray,
                            angle_threshold: float = 0.35,
                            ts: float = 0.005,
                            step_size: int = 100):
        '''
        Replacement CheckIf_LooseSuddenPitch, CheckIf_LooseSuddenRoll
        This is used to detect the pitch or roll angle changes more than 20
        degrees in a ONE SECOND time window. Added heuristic: I created a
        stepping size of (`step_size`), by default, 100 (500 ms, or Nyquist
        Frequency of IMU), and we saw a 10-50x performance gain and same
        evaluation.
        summing between signal[n-200:n] --> low pass filter then integrate
        --> Can make this fast. (It's currently n^2)
        --> can be O(N)
        '''
        for n in range(200, len(signal), step_size):
            delta = np.sum(signal[n-200:n]) * ts
            if np.abs(delta) > angle_threshold:
                return True
        return False

    def check_roll_orientation(self,
                               angle_threshold: float = 0.35,
                               lateral_acc_threshold: float = 1.5,
                               sample_freq: float = 200.0) -> bool:
        '''
        Detect if there is a large persisting offset in acc_y that suggests a
        greater than 20 degree roll angle. v3 only. Make sure that this is not
        caused by turning. Check if angle change is less than 20 degrees.
        At 10mps a 1.5mps2 lat accel means a radius of v2/alat = 70m. Which 10s
        or 100m is more than 1 rad.  Threshold will be at 0.35 rad.
        Notes: Short Burst Only: 10-20 seconds. Notice lateral acc's threshold
        is roughly 10-15% of gravity.
        [DEBUG]:
        Return Turn Angle + Lateral Acceleration (median)
        '''
        turnAng = np.abs(np.sum(self.gyro[:, 3]) / sample_freq)
        if (np.abs(np.median(self.acc[:, 2])) > lateral_acc_threshold) & \
                (turnAng < angle_threshold):
            return True
        return False

    def yaw_jitter(self,
                   hf_yaw_std_threshold: float = 0.1,
                   window_size: float = 2.0,
                   n1: int = 0,
                   sampling_freq: int = 200) -> bool:
        '''
        Assess if the amount of high frequency yaw angle exceeds what is
        reasonable for a sedan.  Trucks in some occasions may show an
        excessively high yaw noise but even there that should be addressed.
        v3 only Look for high power above 2Hz in yaw as an indication that the
        device is loosely mounted in the bracket.
        hpf2hz_gz is "High Pass Filter at 2 Hertz of Yaw Rate" (gz)
        :hf_yaw_std_threshold:
        0.1 for normal vehicles
        0.2 for trucks or highly resonating functions
        :param window_size: window size in seconds. By default it's two seconds
        window length and 200 steps of iteration.
        # DEBUG:
        1. HPF2h_GZ_STD
        '''
        hpf2hz_gz = filtfilt(*butter_highpass(2, 200, 4),
                                       self.gyro[:, 3])
        # Find largest power over a jumping 2s window w 0.5s jumps.
        length = int(window_size * sampling_freq)
        n2 = n1 + length
        while (n2 < len(self.gyro[:, 3]) - length):
            hpf2hz_gz_std = np.std(hpf2hz_gz[n1:n2])
            if (hpf2hz_gz_std > hf_yaw_std_threshold):
                return True
            n1 += 100
            n2 += 100
        return False

    def check_resonance_channels(self,
                                 resonance_count_threshold: int = 3,
                                 gyro_threshold: float = 0.05,
                                 acc_threshold: float = 0.5) -> bool:
        '''
        For now, this will be the main function for chechking the resonance
        channels in the IMU signals.
        [TODO]: Vectorize this classifier so that we don't have to do so
        many copying. Similar to Decay Oscillation, we can apply Wavelet
        filters to
        '''
        idx, cond1_6 = 0, 0
        b, a = butter_lowpass(10, 200, 1)
        gyro_lb = filtfilt(b, a, self.gyro[:, 1:], axis=0)
        acc_lb = filtfilt(b, a, self.acc[:, 1:], axis=0)
        while (cond1_6 < resonance_count_threshold) & (idx <= 5):
            if idx < resonance_count_threshold:
                cond1_6 += self.ResonanceDetected(self.gyro[:, 0] / 1e9,
                                                  gyro_lb[:, idx],
                                                  gyro_threshold)
            else:
                cond1_6 += self.ResonanceDetected(self.acc[:, 0] / 1e9,
                                                  acc_lb[:, idx-3],
                                                  acc_threshold)
            idx += 1

        if cond1_6 >= resonance_count_threshold:
            return True

        return False

    @staticmethod
    def ResonanceDetected(time: np.ndarray,
                          sig: np.ndarray,
                          std_threshold: float,
                          index_offset: int = 400,
                          coeff_vari: float = 0.35,
                          tdelta_lb: float = 0.025,
                          tdelta_ub: float = 0.3
                          ) -> bool:
        '''
        Assess if resonance is observed within a channel. Resonannce is
        defined as a single frequency of sufficient magnitude. (.05)
        This is a helper function (private func) for check_resonance_channel
        for now.
        Notes:
        -599 is to match to look back >3 minutes
        -600 if >= 3 minutes.
        [OD: Erwin wants to choose relative extremas that are past the 5 seconds
        past the initial peak.]
        dt_med is a time delta median -- this is used so that
        reson_stat1 -- coefficient_variance (std / mean)
        [DEBUG]:
        1. ResonStat1: Coefficient Variance
        2. DVAL: Dispersion of Power
        3. dt_med: the change in power.
        Args:
            1. time: sensor_ns in seconds (divide by 1e9)
            2. sig: individual signal passed in by `check_resonance_channel`
            3. std_threshold: the dispersion of power burst from signal
            threshold
            4. index_offset: Erwin looks like to look back two seconds or 400
            samples
            5. coeff_vari: coefficient of variation (see the)
        '''
        peaks = np.c_[argrelextrema(sig[index_offset:len(sig)-1],
                                              np.less),
                      argrelextrema(sig[index_offset:len(sig)-1],
                                              np.greater)
                      ][0] + index_offset
        peaks.sort(kind='merge')
        timePeaks = time[peaks]
        valPeaks = sig[peaks]

        for npk in range(peaks.size):
            if ((timePeaks[npk] - timePeaks[0]) >= 3.0):
                npk_back = find_closest_index(peaks, peaks[npk] - 599, min)
                time_dt = np.diff(timePeaks[npk_back:npk+1])
                dt_med = np.median(time_dt)
                dval_std = np.std(np.diff(valPeaks[npk_back:npk+1]))
                # Measure consistency of time between consecutive peaks are.
                reson_stat1 = np.std(time_dt) / dt_med
                if ((reson_stat1 < coeff_vari) &
                    (dval_std >= std_threshold) &
                        (tdelta_lb <= dt_med <= tdelta_ub)):
                    return True

        return False

    @staticmethod
    def _NumEvidenceDecayingOscillation(sig: np.ndarray,
                                        consecutive_peak_count: int = 5,
                                        min_peak_thres: float = 0.05,
                                        index_offset: int = 1) -> bool:
        """
        `NumEvidenceDecayingOscillation` looks through an entire event
        and checks whether or not the event has 5 consecutive decay
        oscillations.
        :param min_peak_thres: This should be changed so that
        [TODO]: It's unnecessary to iterate through `peaks` two times.
        It might be easier to use a parabola and check if the
        decaying oscillations match this parabola (above a certain threshold)
        of course.
        [IDEA]: Since decay oscillations mimic a decaying envelope, we can
        leverage this. see `scipy.signal.hilbert` documentation online
        [DEBUG]:
        return Counts of Peaks (positive and negative peaks)
        """
        peaks = np.c_[argrelextrema(sig[index_offset:len(sig)-1],
                                              np.greater),
                      argrelextrema(sig[1:len(sig)-1],
                                              np.less)
                      ][0] + index_offset
        peaks.sort(kind='merge')
        valPeaks = sig[peaks]

        # Positive Peaks
        count, prevVal = 0, 0
        for npk in range(peaks.size):
            if (valPeaks[npk] > 0.0):
                if ((valPeaks[npk] < prevVal) &
                        (np.abs(valPeaks[npk]) > min_peak_thres)):
                    count += 1
                    if (count >= consecutive_peak_count):
                        return True
                else:
                    count = 0
                prevVal = valPeaks[npk]

        # Neg peaks
        count, prevVal = 0, 0
        for npk in range(peaks.size):
            if (valPeaks[npk] < 0.0):
                if ((valPeaks[npk] > prevVal) &
                        (np.abs(valPeaks[npk]) > min_peak_thres)):
                    count += 1
                    if (count >= consecutive_peak_count):
                        return True
                else:
                    count = 0
                prevVal = valPeaks[npk]

        return False

    def detect_decaying_oscillation(self) -> bool:
        '''
        Detect dying oscillation as occurs when it falls off the windshield.
        This shows up in GYR unfiltered. Find evidence of decaying oscillation
        in at least one of the three gyro channels.
        Unfortunately, the way the `_NumEvidenceDecayingOscillation`, we have
        to pass in each gyro channel sequentially to evaluate the signal. This
        isn't ideal as we make at most three copies of `gyro_lp`in order get
        results.
        [DEBUG]: Number of Decay Oscillations
        '''
        gyro_lp = filtfilt(*butter_lowpass(10, 200, 1),
                                     self.gyro[:, 1:],
                                     axis=0)

        return ((self._NumEvidenceDecayingOscillation(gyro_lp[:, 0])) |
                (self._NumEvidenceDecayingOscillation(gyro_lp[:, 1])) |
                (self._NumEvidenceDecayingOscillation(gyro_lp[:, 2])))

    def serial_run(self) -> Dict[str, bool]:
        '''
        Evaluate Sensor Files
        [TODO]: Instead of passing default parameters per each detector,
        allow the function to be
        '''
        return {'roll_orientation': self.check_roll_orientation(),
                'pitch_orientation': self.check_pitch_angle(),
                'sudden_roll': self.sudden_angle_change(self.gyro[:, 1]),
                'sudden_pitch': self.sudden_angle_change(self.gyro[:, 2]),
                'yaw_jitter': self.yaw_jitter(),
                'decay_oscillation': self.detect_decaying_oscillation(),
                'loose_resonance': self.check_resonance_channels()
                }

class AtypicalDetectors:

    def __init__(self, sdata: RawIMUPreprocess):
        self.imu = sdata
        if isinstance(self.imu, RawIMUPreprocess):
            assert self.imu.angle is not None
            assert self.imu.acc is not None
            assert self.imu.gyro is not None
        elif isinstance(sdata, list):
            self.imu = RawIMUPreprocess(sdata, orient_sensor=True)


    def check_pitch_angle(self) -> bool:
        '''Replacement for CheckIf_LooseOri
        [DEBUG]:
        return `angle`
        '''
        return False if 115.0 <= self.imu.angle <= 160.0 else True

    @staticmethod
    def sudden_angle_change(signal: np.ndarray,
                            angle_threshold: float = 0.35,
                            ts: float = 0.005,
                            step_size: int = 100):
        '''
        Replacement CheckIf_LooseSuddenPitch, CheckIf_LooseSuddenRoll
        This is used to detect the pitch or roll angle changes more than 20
        degrees in a ONE SECOND time window. Added heuristic: I created a
        stepping size of (`step_size`), by default, 100 (500 ms, or Nyquist
        Frequency of IMU), and we saw a 10-50x performance gain and same
        evaluation.
        summing between signal[n-200:n] --> low pass filter then integrate
        '''
        for n in range(200, len(signal), step_size):
            delta = np.sum(signal[n-200:n]) * ts
            if np.abs(delta) > angle_threshold:
                return True
        return False

    def check_roll_orientation(self,
                               angle_threshold: float = 0.35,
                               lateral_acc_threshold: float = 1.5,
                               sample_freq: float = 200.0) -> bool:
        '''
        Detect if there is a large persisting offset in acc_y that suggests a
        greater than 20 degree roll angle. v3 only. Make sure that this is not
        caused by turning. Check if angle change is less than 20 degrees.
        At 10mps a 1.5mps2 lat accel means a radius of v2/alat = 70m. Which 10s
        or 100m is more than 1 rad.  Threshold will be at 0.35 rad.
        Notes: Short Burst Only: 10-20 seconds. Notice lateral acc's threshold
        is roughly 10-15% of gravity.
        [DEBUG]:
        Return Turn Angle + Lateral Acceleration (median)
        '''
        turnAng = np.abs(np.sum(self.imu.gyro[:, 3]) / sample_freq)
        if (np.abs(np.median(self.imu.acc[:, 2])) > lateral_acc_threshold) & \
                (turnAng < angle_threshold):
            return True
        return False

    def yaw_jitter(self,
                   hf_yaw_std_threshold: float = 0.1,
                   window_size: float = 2.0,
                   n1: int = 0,
                   sampling_freq: int = 200) -> bool:
        '''
        Assess if the amount of high frequency yaw angle exceeds what is
        reasonable for a sedan.  Trucks in some occasions may show an
        excessively high yaw noise but even there that should be addressed.
        v3 only Look for high power above 2Hz in yaw as an indication that the
        device is loosely mounted in the bracket.
        hpf2hz_gz is "High Pass Filter at 2 Hertz of Yaw Rate" (gz)
        :hf_yaw_std_threshold:
        0.1 for normal vehicles
        0.2 for trucks or highly resonating functions
        :param window_size: window size in seconds. By default it's two seconds
        window length and 200 steps of iteration.
        # DEBUG:
        1. HPF2h_GZ_STD
        '''
        hpf2hz_gz = filtfilt(*butter_highpass(2, 200, 4),
                                       self.imu.gyro[:, 3])
        # Find largest power over a jumping 2s window w 0.5s jumps.
        length = int(window_size * sampling_freq)
        n2 = n1 + length
        while (n2 < len(self.imu.gyro[:, 3]) - length):
            hpf2hz_gz_std = np.std(hpf2hz_gz[n1:n2])
            if (hpf2hz_gz_std > hf_yaw_std_threshold):
                return True
            n1 += 100
            n2 += 100
        return False

    def check_resonance_channels(self,
                                 resonance_count_threshold: int = 3,
                                 gyro_threshold: float = 0.05,
                                 acc_threshold: float = 0.5) -> bool:
        '''
        For now, this will be the main function for chechking the resonance
        channels in the IMU signals.
        [TODO]: Vectorize this classifier so that we don't have to do so
        many copying. Similar to Decay Oscillation, we can apply Wavelet
        filters to
        '''
        idx, cond1_6 = 0, 0
        b, a = butter_lowpass(10, 200, 1)
        gyro_lb = filtfilt(b, a, self.imu.gyro[:, 1:], axis=0)
        acc_lb = filtfilt(b, a, self.imu.acc[:, 1:], axis=0)
        while (cond1_6 < resonance_count_threshold) & (idx <= 5):
            if idx < resonance_count_threshold:
                cond1_6 += self.ResonanceDetected(self.imu.gyro[:, 0] / 1e9,
                                                  gyro_lb[:, idx],
                                                  gyro_threshold)
            else:
                cond1_6 += self.ResonanceDetected(self.imu.acc[:, 0] / 1e9,
                                                  acc_lb[:, idx-3],
                                                  acc_threshold)
            idx += 1

        if cond1_6 >= resonance_count_threshold:
            return True

        return False

    @staticmethod
    def ResonanceDetected(time: np.ndarray,
                          sig: np.ndarray,
                          std_threshold: float,
                          index_offset: int = 400,
                          coeff_vari: float = 0.35,
                          tdelta_lb: float = 0.025,
                          tdelta_ub: float = 0.3
                          ) -> bool:
        '''
        Assess if resonance is observed within a channel. Resonannce is
        defined as a single frequency of sufficient magnitude. (.05)
        This is a helper function (private func) for check_resonance_channel
        for now.
        Notes:
        -599 is to match to look back >3 minutes
        -600 if >= 3 minutes.
        [OD: Erwin wants to choose relative extremas that are past the 5 seconds
        past the initial peak.]
        dt_med is a time delta median -- this is used so that
        reson_stat1 -- coefficient_variance (std / mean)
        [DEBUG]:
        1. ResonStat1: Coefficient Variance
        2. DVAL: Dispersion of Power
        3. dt_med: the median change in power.
        Args:
            1. time: sensor_ns in seconds (divide by 1e9)
            2. sig: individual signal passed in by `check_resonance_channel`
            3. std_threshold: the dispersion of power burst from signal
            threshold
            4. index_offset: Erwin looks like to look back two seconds or 400
            samples
            5. coeff_vari: coefficient of variation (see the)
        '''
        peaks = np.c_[argrelextrema(sig[index_offset:len(sig)-1],
                                              np.less),
                      argrelextrema(sig[index_offset:len(sig)-1],
                                              np.greater)
                      ][0] + index_offset
        peaks.sort(kind='merge')
        timePeaks = time[peaks]
        valPeaks = sig[peaks]

        for npk in range(peaks.size):
            if ((timePeaks[npk] - timePeaks[0]) >= 3.0):
                npk_back = find_closest_index(peaks, peaks[npk] - 599, min)
                time_dt = np.diff(timePeaks[npk_back:npk+1])
                dt_med = np.median(time_dt)
                dval_std = np.std(np.diff(valPeaks[npk_back:npk+1]))
                # Measure consistency of time between consecutive peaks are.
                reson_stat1 = np.std(time_dt) / dt_med
                if ((reson_stat1 < coeff_vari) &
                    (dval_std >= std_threshold) &
                    (tdelta_lb <= dt_med <= tdelta_ub)):
                    return True

        return False

    @staticmethod
    def _NumEvidenceDecayingOscillation(sig: np.ndarray,
                                        consecutive_peak_count: int = 5,
                                        min_peak_thres: float = 0.05,
                                        index_offset: int = 1) -> bool:
        """
        `NumEvidenceDecayingOscillation` looks through an entire event
        and checks whether or not the event has 5 consecutive decay
        oscillations.
        :param min_peak_thres: This should be changed so that
        [TODO]: It's unnecessary to iterate through `peaks` two times.
        It might be easier to use a parabola and check if the
        decaying oscillations match this parabola (above a certain threshold)
        of course.
        [IDEA]: Since decay oscillations mimic a decaying envelope, we can
        leverage this. see `scipy.signal.hilbert` documentation online
        [DEBUG]:
        return Counts of Peaks (positive and negative peaks)
        """
        peaks = np.c_[argrelextrema(sig[index_offset:len(sig)-1],
                                              np.greater),
                      argrelextrema(sig[1:len(sig)-1],
                                              np.less)
                      ][0] + index_offset
        peaks.sort(kind='merge')
        valPeaks = sig[peaks]

        # Positive Peaks
        count, prevVal = 0, 0
        for npk in range(peaks.size):
            if (valPeaks[npk] > 0.0):
                if ((valPeaks[npk] < prevVal) &
                        (np.abs(valPeaks[npk]) > min_peak_thres)):
                    count += 1
                    if (count >= consecutive_peak_count):
                        return True
                else:
                    count = 0
                prevVal = valPeaks[npk]

        # Neg peaks
        count, prevVal = 0, 0
        for npk in range(peaks.size):
            if (valPeaks[npk] < 0.0):
                if ((valPeaks[npk] > prevVal) &
                        (np.abs(valPeaks[npk]) > min_peak_thres)):
                    count += 1
                    if (count >= consecutive_peak_count):
                        return True
                else:
                    count = 0
                prevVal = valPeaks[npk]

        return False

    def detect_decaying_oscillation(self) -> bool:
        '''
        Detect dying oscillation as occurs when it falls off the windshield.
        This shows up in GYR unfiltered. Find evidence of decaying oscillation
        in at least one of the three gyro channels.
        Unfortunately, the way the `_NumEvidenceDecayingOscillation`, we have
        to pass in each gyro channel sequentially to evaluate the signal. This
        isn't ideal as we make at most three copies of `gyro_lp`in order get
        results.
        [DEBUG]: Number of Decay Oscillations
        '''
        gyro_lp = filtfilt(*butter_lowpass(10, 200, 1),
                                     self.imu.gyro[:, 1:],
                                     axis=0)

        return ((self._NumEvidenceDecayingOscillation(gyro_lp[:, 0])) |
                (self._NumEvidenceDecayingOscillation(gyro_lp[:, 1])) |
                (self._NumEvidenceDecayingOscillation(gyro_lp[:, 2])))

    def serial_run(self) -> Dict[str, bool]:
        '''
        Evaluate Sensor Files
        [TODO]: Instead of passing default parameters per each detector,
        allow the function to be
        '''
        return {'roll_orientation': self.check_roll_orientation(),
                'pitch_orientation': self.check_pitch_angle(),
                'sudden_roll': self.sudden_angle_change(self.imu.gyro[:, 1]),
                'sudden_pitch': self.sudden_angle_change(self.imu.gyro[:, 2]),
                'yaw_jitter': self.yaw_jitter(),
                'decay_oscillation': self.detect_decaying_oscillation(),
                'loose_resonance': self.check_resonance_channels()
                }