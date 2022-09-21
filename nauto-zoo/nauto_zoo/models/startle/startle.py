import numpy as np
import scipy as sp
from scipy.signal import butter

import logging
from sensor import sensor_pb2
from nauto_datasets.utils import protobuf
from nauto_datasets.core.sensors import CombinedRecording, Recording
from pathlib import Path

from collections import OrderedDict
from nauto_zoo import DoNotWantToProduceJudgementError, MalformedModelInputError, Timeline, TimelineElement


def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog=False)
    return b, a


def DetectStartle(time, lpf10_accx, lpf20_accx):
    """Core startle detection function"""
    a = lpf10_accx

    # Set up a series of 50ms sliding windows that maintain 1st moments to compute mean.
    N = len(a);
    M1Wins = np.zeros(4);  # 4th window is 30ms
    MeanWins = np.zeros(4);
    tmparr = np.zeros((4, N));
    K = 5;  # 200Hz to get 50ms is 10 samples.
    halfK = int((K - 1) / 2);
    # Expect no more than 1000 panic sampels within 10s of data. If that happens then crazy noise.
    panicTimes = np.zeros(1000);
    panicIndices = np.zeros(1000).astype(int);
    panicSeverity = np.zeros(1000);
    panicTrue = 0 * panicIndices;
    TruePanicIndices = np.zeros(1000).astype(int);  # Panic location indices after duration requirement logic applied.

    numPanics = 0;
    cnt = 0;  # Number of samples in buffer to make sure initialization works well.
    for n in np.arange(3 * K, N - 3 * K, 1):  # Loop through all data
        for wi in np.arange(0, 4, 1):  # 4 windows of 25ms (5 samples) each.
            if (wi <= 2):
                halfwi = 1;  # (wimax-1)/2
                wStartIndex = (wi - halfwi) * K;
                if (cnt >= K):
                    M1Wins[wi] = M1Wins[wi] + a[n - wStartIndex] - a[n - wStartIndex - K];
                else:
                    M1Wins[wi] = M1Wins[wi] + a[n - wStartIndex];
                MeanWins[wi] = M1Wins[wi] / K;
                tmparr[wi, n] = MeanWins[wi];
                # print(n,wi,-wStartIndex - K,-wStartIndex)
            if (wi == 3):
                # Select a window of K width that lies ahead of wi=0
                halfwi = 1;  # (wimax-1)/2
                wStartIndex = (-1 - halfwi) * K;
                if (cnt >= K):
                    M1Wins[wi] = M1Wins[wi] + a[n - wStartIndex] - a[n - wStartIndex - K];
                else:
                    M1Wins[wi] = M1Wins[wi] + a[n - wStartIndex];
                MeanWins[wi] = M1Wins[wi] / K;
                tmparr[wi, n] = MeanWins[wi];
                # print(n,wi,-wStartIndex - K,-wStartIndex)
        cnt = cnt + 1;
        # Check if a panic decel occured
        # index 0 is ahead of index 2 in time.   The MeanWins[3] assures that deceleration persists.
        # panic = ((MeanWins[2] - MeanWins[0]) > 2.0) & (MeanWins[2] < 1.0) & (MeanWins[0] < -1.0) & (MeanWins[3] < -2.0); # Has to be a decel
        #          change more than 2 towards decel .       deceleration persists.
        panic = ((MeanWins[2] - MeanWins[0]) > 2.0) and \
                ((MeanWins[2] - MeanWins[3]) > 2.0) and \
                (MeanWins[3] < -1.0);  # Has to be a decel
        if (panic):
            panicTimes[numPanics] = time[n - halfK];
            panicIndices[numPanics] = n - halfK;
            panicSeverity[numPanics] = MeanWins[3] - MeanWins[2];
            numPanics = numPanics + 1;

    # Below is the logic to filter out false panics (e.g. speed bumps).  These are of a shorter
    # duration than what humans produce - human reaction time is 300ms so it is expected that an
    # intentional human action lasts at least 300ms.

    PANIC = False;
    if (len(panicTimes) > 0):  # always True
        panicDecelerations = a[panicIndices];
        # Assess if a true panic
        # Need to have at least 4 panic times within 30ms (ideally 5 within 25ms);
        if (len(panicTimes) >= 4):
            for pi in range(len(panicTimes) - 3):
                if (((panicTimes[pi + 3] - panicTimes[pi]) < 0.03) and
                        (panicDecelerations[pi + 3] < panicDecelerations[pi])):
                    panicTrue[pi:pi + 3 + 1] = 1;
                    PANIC = True;

    BUMP = False;
    if (PANIC):

        # Check the 20Hz LPF and for each panic potential walk forward and back and remove those
        # where the same decel is reached within 200ms before or after.
        for j in range(len(panicTimes)):
            if (panicTrue[j] == 1):
                nj = panicIndices[j];
                refaccel = lpf20_accx[nj];
                if (refaccel < 0.0):  # These are always on a down slope and thus only walking forward makes sense.
                    # Walk forward to where it transitions same ref value again
                    nStart = nj;
                    nn = nj + 1;
                    while (lpf20_accx[nn] <= refaccel):
                        nn = nn + 1;
                        if (nn >= N - 1):
                            break;
                    nEnd = nn - 1;  # Last one exceeded the threshold.
                    if ((time[nEnd] - time[nStart]) < 0.15):
                        panicTrue[j] = -1;

    # Expect not more than 100 panics in one 10s window.
    ExpectedBumpEndTimes = np.zeros(100);
    ExpectedBumpStartTimes = np.zeros(100);
    BumpPanicTimes = np.zeros(100);
    Num_ExpectedBumpStartTimes = 0;

    # Check if the 4m/s2 decel period lasts less than 300ms.  For a human intended decel it lasts at
    # least 300ms because otherwise there is not enough time to initialize and cancel it.
    # These fast ones are expected to all be related to bumps.
    # For each panic event, walk forward if greater than 4mps2 or back otherwise and assess if the duration
    # of decel exceeding 4mps2 is less than 300ms.  If so make the entire exterbal video redish.

    for j in range(len(panicTimes)):
        if (np.absolute(panicTrue[j]) > 0.01):  # Make sure that we look at the bumps for now to get all detectors.
            nj = panicIndices[j];
            if (lpf10_accx[nj] <= -1.0):  # Make sure panic is on decel portion.
                # Walk forward to where it transitions -2mps2 adnl decel
                refDecel = lpf10_accx[nj];
                # Check if refDecel within 2mps2 from peak that occurs within 1s from current time.
                # Otherwise likely that it will last less than 300ms.
                peakDecel = np.min(lpf10_accx[nj:np.min([nj + 200, len(lpf10_accx)])]);
                if (refDecel - 2.0 < peakDecel + 2.0):
                    continue;
                nn = nj;
                while (lpf10_accx[nn] >= refDecel - 2.0):
                    nn = nn + 1;
                    if (nn >= N - 1):
                        break;
                nStart = nn + 1;
                # Continue to walk to see when it drops to less decel than that again.
                nn = nStart;
                while (lpf10_accx[nn] <= refDecel - 2.0):
                    nn = nn + 1;
                    if (nn >= N - 1):
                        break;
                nEnd = nn - 1;
                if ((time[nEnd] - time[nStart]) < 0.3):
                    ExpectedBumpEndTimes[Num_ExpectedBumpStartTimes] = time[nEnd];
                    ExpectedBumpStartTimes[Num_ExpectedBumpStartTimes] = time[nStart];
                    BumpPanicTimes[Num_ExpectedBumpStartTimes] = panicTimes[j];
                    # ExpectedBumpStartTimes[Num_ExpectedBumpStartTimes] = panicTimes[j];
                    Num_ExpectedBumpStartTimes = Num_ExpectedBumpStartTimes + 1;
                    # Color ext video red.
                    BUMP = True;

    TRUE_PANIC_FOUND = False;
    for j in range(len(panicTimes) - 3):
        if (np.sum(panicTrue[j:j + 4]) > 3.9):
            # A panic requires that at least 4 panic estimates in sucession are true (i.e.
            # rapid change in decel occurs for at least 20ms).
            # Rate the panic severity and show for severe ones what the magnitude was (max over these 4 samples).
            # A severity of 3 or greater is generally associated with facial expresions that
            # suggest surprise and/or fear.
            # ax.plot(panicTimes[j:j+4],lpf10_accx[panicIndices[j:j+4]],'kd',MarkerFaceColor='r',MarkerSize=5)
            # plt.text(panicTimes[j+1],lpf10_accx[panicIndices[j+1]], '%.2f ' % np.max(panicSeverity[j:j+4]),
            #     bbox=dict(facecolor='yellow', alpha=0.8))
            TruePanicIndices[j:j + 4] = panicIndices[j:j + 4];
            TRUE_PANIC_FOUND = True;
            # break; # for now only show the first time panic is detected.

    return PANIC, BUMP, panicTrue[0:numPanics], panicTimes[0:numPanics], panicIndices[0:numPanics], panicSeverity[
                                                                                                    0:numPanics], \
           Num_ExpectedBumpStartTimes, ExpectedBumpStartTimes[0:Num_ExpectedBumpStartTimes], \
           ExpectedBumpEndTimes[0:Num_ExpectedBumpStartTimes], BumpPanicTimes[0:Num_ExpectedBumpStartTimes], \
           TRUE_PANIC_FOUND, TruePanicIndices[0:numPanics];


#
# Processing functions
#

def get_startle(time, accx):
    """Call startle detection and format output"""
    # filtered signals
    [b, a] = butter_lowpass(10, 200, 4);
    lpf10_accx = sp.signal.filtfilt(b, a, accx);
    [b, a] = butter_lowpass(20, 200, 4);
    lpf20_accx = sp.signal.filtfilt(b, a, accx)

    # panic events
    PANIC, BUMP, panicTrue, panicTimes, panicIndices, panicSeverity, Num_ExpectedBumpStartTimes, \
    ExpectedBumpStartTimes, ExpectedBumpEndTimes, BumpPanicTimes, \
    TRUE_PANIC_FOUND, TruePanicIndices = DetectStartle(time, lpf10_accx, lpf20_accx);

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


def get_consecutive_time_segments(times, indices):
    """Find consecutive time segments based on indices"""
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


def sec_to_sensor_ns(tarr, offset_ns=0):
    return (np.int64(np.array(tarr) * 1e9 + offset_ns)).tolist()


def process_one_event(com_rec: CombinedRecording) -> Timeline:
    imu = get_sensors_from_combined_recordings(com_rec)
    rec = OrderedDict({
        'startle_times_ns': [],
        'startle_severity': [],
        'sensor_offset_ns': None
    })

    # extract sensor data
    sensor_offset_ns = int(imu['sensor_ns'].min())
    time_s = (imu['sensor_ns'] - sensor_offset_ns)*1e-9
    acc_x = imu['acc_x']
    if time_s.shape != acc_x.shape:
        raise MalformedModelInputError('Signals data does not match')
    # sort by time
    sort_idx = np.argsort(time_s)
    time_s = time_s[sort_idx]
    acc_x = acc_x[sort_idx]
    # get startle brakes
    startle_times, startle_severity = get_startle(time_s, acc_x)
    # convert to sensor time ns
    startle_times_ns = sec_to_sensor_ns(startle_times,sensor_offset_ns)

    timeline = Timeline(review_type="startle_times_ns")
    for i in range(0, len(startle_times_ns)):
        timeline.add_element(TimelineElement(
            start_ns=startle_times_ns[i][0],
            end_ns=startle_times_ns[i][1],
            element_type='startle_severity',
            extra_fields={
                "value": startle_severity[i]
            }
        ))
    timeline.set_offset_ns(sensor_offset_ns)

    return timeline


def get_sensors_from_combined_recordings(com_rec: CombinedRecording):
    if not hasattr(com_rec, 'oriented_acc'):
        raise DoNotWantToProduceJudgementError()

    acc_orig = com_rec.oriented_acc.stream._asdict()
    # gyro = com_rec.oriented_gyro.stream._asdict()
    # gps = com_rec.gps.stream._asdict()

    if 'sensor_ns' not in acc_orig.keys() or len(acc_orig['sensor_ns']) < 1:
        raise ValueError('Sensor records do not have acc data.')

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
