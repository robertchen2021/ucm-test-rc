# python 3.6

from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter


##############################
# This file contains some IMU-based algorithms
# Primary author: Erwin Boer
##############################

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog=False)
    return b, a


def ComputeBrakeSeverity(time, decel):
    # Return array with brake severity computed for every sample.
    N = len(time)
    BrakeSeverity = 0 * time
    for n in range(N):
        bsBest = 0.0
        nBest = n
        for nn in range(n + 1, N):
            bs = np.sign(decel[nn]) * np.square(decel[nn] - decel[n]) / (time[nn] - time[n])
            if (bs < bsBest) & (decel[nn] < decel[n]):  # Assure that brake severity only computed for decelerations.
                # if (bs < bsBest):
                bsBest = bs
                nBest = nn  # Indicates the end of a brake onset.
            if ((time[nn] - time[n]) > 0.25) & (nn != nBest):  # Only look 250ms ahead unless the brake severity is still growing.
                break
        if (BrakeSeverity[nBest] > bsBest):
            BrakeSeverity[nBest] = bsBest
    return BrakeSeverity


def DetectSpeedBump_old(time, accx, gpsspeed):  # ,tit):

    # Determine if a speed bump was found. If so, assume that any brake severity
    # within 0.3s before and 2.0s after are speed bump related an not actual hare braking.  These timing
    # checks are performed in function FindPanicBrakingEvents.  In the current function, only bump detection
    # is performed.

    # orig: GPS at 1Hz, IMU at 200Hz
    # here: IMU and GPS at 200 Hz (no gpstime)

    # Filter accx w 10Hz lpf
    [b, a] = butter_lowpass(10, 200, 4)
    sig = sp.signal.filtfilt(b, a, accx)
    N = len(sig)

    peaks = np.zeros(N)  # Not likely to have that many but just in case of crazy noise.
    peaktimes = np.zeros(N)
    peakindices = np.zeros(N).astype(int)
    peakspeeds = np.zeros(N)  # Speed at which the peak occurs - needed to set
    # time spacing between peaks based on assumed 2m wheel base
    numpeaks = 0

    # Find peaks in sig
    for n in np.arange(1, N - 1):
        # Pos peak
        if ((sig[n] > sig[n - 1]) & (sig[n] > sig[n + 1])):
            peaks[numpeaks] = sig[n]
            peaktimes[numpeaks] = time[n]
            peakindices[numpeaks] = n
            # Find nearest time in gps and set peak speed.
            # i = np.argmin(np.absolute(gpstime - time[n]))
            peakspeeds[numpeaks] = gpsspeed[n]
            numpeaks = numpeaks + 1

        # Neg peak
        if ((sig[n] < sig[n - 1]) and (sig[n] < sig[n + 1])):
            peaks[numpeaks] = sig[n]
            peaktimes[numpeaks] = time[n]
            peakindices[numpeaks] = n
            # Find nearest time in gps and set peak speed.
            # i = np.argmin(np.absolute(gpstime - time[n]))
            peakspeeds[numpeaks] = gpsspeed[n]
            numpeaks = numpeaks + 1

    # Detect if a positive and negative accx bump are of similar height and within wheel axis / speed seconds
    wheelbase = 2.0  # Assumed for all vehicle for now.
    DETECTED_BUMP = False
    bumpindices = np.zeros(N).astype(int)
    numbumps = 0
    for k in np.arange(2, numpeaks):
        # if ( (peaks[k-1] < -2.0) & (peaks[k] > 2.0) & ((peaktimes[k] - peaktimes[k-1]) < (wheelbase/peakspeeds[k-1] + 0.1)) ):
        if ((peaks[k - 1] < -1.5) and (peaks[k] > 1.0) and
                (peakspeeds[k - 1] > 0) and
                ((peaktimes[k] - peaktimes[k - 1]) < (wheelbase / peakspeeds[k - 1] + 0.3))):
            # We need to make sure that the peak before the 1st peak is at least twice as small in magnitude.
            if (peaks[k - 2] < 0.5 * peaks[k - 1]):
                continue
            # Make sure that the speed is not nearly zero at the 2nd peak.
            # We need to avoid the situation where the car comes to a stand still and bounces in its suspension.
            # Check if the speed 2s later is zero.  If so, then it is not a bump.  Of course possible that
            # brake on a bump nearly to zero speed but that is quite unlikely.
            n_check = peakindices[k] + 200
            if (n_check < len(time)):
                # i = np.argmin(np.absolute(gpstime - time[n_check])) # Find nearest GPS index at possible speed bump time.
                if (gpsspeed[n_check] > 0.5):
                    bumpindices[numbumps] = peakindices[k - 1]
                    numbumps = numbumps + 1
                    DETECTED_BUMP = True

                    # Boolean if any bumps are detected.  If so, bump indices holds indices in original time and accx
    # where the bumps was detected i.e. location of first negative peak.
    return DETECTED_BUMP, bumpindices[0:numbumps]


def FindPanicBrakingEvents(file_index, TimeOffset, dev_id, mes_id, time, accx, gpsspeed):
    # Return panic braking events (i.e. brake severity exceeding -200).
    # Input is raw oriented imu accx at 200Hz and raw GPS at 1Hz.
    # Note that time and gpstime are number of seconds from start of event (i.e. 1st sample in IMU accx data).
    # The TimeOffset is the 1st time in the IMU accx sample in seconds.  This allows for conversion back to original nanoseconds.

    BSdf = []  # If no sufficiently severy braking was found, it returns -1.  Otherwise a data structure as indiocated below.

    [b, a] = butter_lowpass(10, 200, 4)
    lpf10_accx = sp.signal.filtfilt(b, a, accx)

    DETECTED_BUMP, bumpindices = DetectSpeedBump_old(time, accx, gpsspeed)  # ,'File %d' % (i))

    # Create a time series of braking severity (delta_mag squared / delta_time).
    BrakeSeverity = ComputeBrakeSeverity(time, lpf10_accx)

    # Given BrakeSeverity and bumpindices create time windows over which the MinBrakeSeverity ess
    # than some threshold.
    # Create a pandas data frame with the panic brake windows.
    # For each brake severity, find start and end of braking.  First check if the brake severity already processed.

    NumPanicEvents = 0  # Number of braking events that have beyond threshold severity and are not speed bumps.
    BrakeSeverityThreshold = -100
    decel = lpf10_accx
    Nbs = len(BrakeSeverity)
    n = 1
    while (n < Nbs):
        # for n in range(1,Nbs):
        if (BrakeSeverity[n] < BrakeSeverityThreshold):
            # Check if time already in an existing BS window.
            # Note that if there are more severe BS within the time window, the max BS value will be stored.
            # Find start and end of brake maneuver.
            # Maintain max BS found within window.
            maxBS = BrakeSeverity[n]
            nn = n
            while (decel[nn - 1] > decel[nn]):
                nn = nn - 1
                if (BrakeSeverity[nn] < maxBS):
                    maxBS = BrakeSeverity[nn]
                if (nn == 0):
                    break
            startTime = time[nn]
            startIndex = nn
            nn = n
            while nn < Nbs and (decel[nn + 1] < decel[nn]):
                nn = nn + 1
                if (BrakeSeverity[nn] < maxBS):
                    maxBS = BrakeSeverity[nn]
                if (nn == Nbs - 1):
                    break
            endTime = time[nn]
            endIndex = nn

            midTime = (startTime + endTime) / 2.0

            # Check if a bump was detected near this panic response. If so, ignore panic.
            KeepBS = True
            if (DETECTED_BUMP):  # Check if a bump occured within speed * delta_time < 2.5m (i.e. wheelbase)
                # print('bumpindices:',len(bumpindices))
                # Reset the panic indices
                for j in range(len(bumpindices)):
                    # Look at speed at bump and then go equiv of 2.5m back in time and see if a panic occured.
                    tbump = time[bumpindices[j]]
                    tbs = midTime
                    if (tbs > (tbump - 0.3)) and (tbs < tbump + 2.0):
                        KeepBS = False
                        # print('BS BeGone')
                    else:
                        # print('BS Keep')
                        pass

            if (KeepBS):
                # We have a new event that we need to store.
                BSdataframe = pd.DataFrame({'FileIndex': [file_index],
                                            'TimeOffset_s': [TimeOffset],  # Offset in seconds
                                            'DeviceID': [dev_id],
                                            'MessageID': [mes_id],
                                            'StartTime_s': [startTime],
                                            # start time in seconds from start of imu timeseries
                                            'EndTime_s': [endTime],
                                            'StartIndex': [startIndex],
                                            'EndIndex': [endIndex],
                                            'BrakeSeverity': [
                                                maxBS]})  # Value (negative) of the maximum brake severity.
                if (NumPanicEvents == 0):
                    BSdf = BSdataframe
                else:
                    BSdf = BSdf.append(BSdataframe, ignore_index=True)
                NumPanicEvents = NumPanicEvents + 1

            # Now we can simply walk forward from the current nn
            n = nn + 1
        else:
            n = n + 1

    return BSdf


def DetectStartle(t, a, lpf20_accx):
    lpf10_accx = a
    time = t

    # Set up a series of 50ms sliding windows that maintain 1st moments to compute mean.
    N = len(a)
    M1Wins = np.zeros(4)  # 4th window is 30ms
    MeanWins = np.zeros(4)
    tmparr = np.zeros((4, N))
    K = 5  # 200Hz to get 50ms is 10 samples.
    halfK = int((K - 1) / 2)
    # Expect no more than 1000 panic sampels within 10s of data. If that happens then crazy noise.
    panicTimes = np.zeros(1000)
    panicIndices = np.zeros(1000).astype(int)
    panicSeverity = np.zeros(1000)
    panicTrue = 0 * panicIndices
    TruePanicIndices = np.zeros(1000).astype(int)  # Panic location indices after duration requirement logic applied.

    numPanics = 0
    cnt = 0  # Number of samples in buffer to make sure initialization works well.
    for n in np.arange(3 * K, N - 3 * K, 1):  # Loop through all data
        for wi in np.arange(0, 4, 1):  # 4 windows of 25ms (5 samples) each.
            if (wi <= 2):
                halfwi = 1  # (wimax-1)/2
                wStartIndex = (wi - halfwi) * K
                if (cnt >= K):
                    M1Wins[wi] = M1Wins[wi] + a[n - wStartIndex] - a[n - wStartIndex - K]
                else:
                    M1Wins[wi] = M1Wins[wi] + a[n - wStartIndex]
                MeanWins[wi] = M1Wins[wi] / K
                tmparr[wi, n] = MeanWins[wi]
                # print(n,wi,-wStartIndex - K,-wStartIndex)
            if (wi == 3):
                # Select a window of K width that lies ahead of wi=0
                halfwi = 1  # (wimax-1)/2
                wStartIndex = (-1 - halfwi) * K
                if (cnt >= K):
                    M1Wins[wi] = M1Wins[wi] + a[n - wStartIndex] - a[n - wStartIndex - K]
                else:
                    M1Wins[wi] = M1Wins[wi] + a[n - wStartIndex]
                MeanWins[wi] = M1Wins[wi] / K
                tmparr[wi, n] = MeanWins[wi]
                # print(n,wi,-wStartIndex - K,-wStartIndex)
        cnt = cnt + 1
        # Check if a panic decel occured
        # index 0 is ahead of index 2 in time.   The MeanWins[3] assures that deceleration persists.
        # panic = ((MeanWins[2] - MeanWins[0]) > 2.0) & (MeanWins[2] < 1.0) & (MeanWins[0] < -1.0) & (MeanWins[3] < -2.0) # Has to be a decel
        #          change more than 2 towards decel .       deceleration persists.
        panic = ((MeanWins[2] - MeanWins[0]) > 2.0) and \
                ((MeanWins[2] - MeanWins[3]) > 2.0) and \
                (MeanWins[3] < -1.0)  # Has to be a decel
        if (panic):
            panicTimes[numPanics] = t[n - halfK]
            panicIndices[numPanics] = n - halfK
            panicSeverity[numPanics] = MeanWins[3] - MeanWins[2]
            numPanics = numPanics + 1

    # Below is the logic to filter out false panics (e.g. speed bumps).  These are of a shorter
    # duration than what humans produce - human reaction time is 300ms so it is expected that an
    # intentional human action lasts at least 300ms.

    PANIC = False
    if (len(panicTimes) > 0):  # always True
        panicDecelerations = a[panicIndices]
        # Assess if a true panic
        # Need to have at least 4 panic times within 30ms (ideally 5 within 25ms)
        if (len(panicTimes) >= 4):
            for pi in range(len(panicTimes) - 3):
                if (((panicTimes[pi + 3] - panicTimes[pi]) < 0.03) and
                        (panicDecelerations[pi + 3] < panicDecelerations[pi])):
                    panicTrue[pi:pi + 3 + 1] = 1
                    PANIC = True

    BUMP = False
    if (PANIC):

        # Check the 20Hz LPF and for each panic potential walk forward and back and remove those
        # where the same decel is reached within 200ms before or after.
        for j in range(len(panicTimes)):
            if (panicTrue[j] == 1):
                nj = panicIndices[j]
                refaccel = lpf20_accx[nj]
                if (refaccel < 0.0):  # These are always on a down slope and thus only walking forward makes sense.
                    # Walk forward to where it transitions same ref value again
                    nStart = nj
                    nn = nj + 1
                    while (lpf20_accx[nn] <= refaccel):
                        nn = nn + 1
                        if (nn >= N - 1):
                            break
                    nEnd = nn - 1  # Last one exceeded the threshold.
                    if ((time[nEnd] - time[nStart]) < 0.15):
                        panicTrue[j] = -1

    # Expect not more than 100 panics in one 10s window.
    ExpectedBumpEndTimes = np.zeros(100)
    ExpectedBumpStartTimes = np.zeros(100)
    BumpPanicTimes = np.zeros(100)
    Num_ExpectedBumpStartTimes = 0

    # Check if the 4m/s2 decel period lasts less than 300ms.  For a human intended decel it lasts at
    # least 300ms because otherwise there is not enough time to initialize and cancel it.
    # These fast ones are expected to all be related to bumps.
    # For each panic event, walk forward if greater than 4mps2 or back otherwise and assess if the duration
    # of decel exceeding 4mps2 is less than 300ms.  If so make the entire exterbal video redish.

    for j in range(len(panicTimes)):
        if (np.absolute(panicTrue[j]) > 0.01):  # Make sure that we look at the bumps for now to get all detectors.
            nj = panicIndices[j]
            if (lpf10_accx[nj] <= -1.0):  # Make sure panic is on decel portion.
                # Walk forward to where it transitions -2mps2 adnl decel
                refDecel = lpf10_accx[nj]
                # Check if refDecel within 2mps2 from peak that occurs within 1s from current time.
                # Otherwise likely that it will last less than 300ms.
                peakDecel = np.min(lpf10_accx[nj:np.min([nj + 200, len(lpf10_accx)])])
                if (refDecel - 2.0 < peakDecel + 2.0):
                    continue
                nn = nj
                while (lpf10_accx[nn] >= refDecel - 2.0):
                    nn = nn + 1
                    if (nn >= N - 1):
                        break
                nStart = nn + 1
                # Continue to walk to see when it drops to less decel than that again.
                nn = nStart
                while (lpf10_accx[nn] <= refDecel - 2.0):
                    nn = nn + 1
                    if (nn >= N - 1):
                        break
                nEnd = nn - 1
                if ((time[nEnd] - time[nStart]) < 0.3):
                    ExpectedBumpEndTimes[Num_ExpectedBumpStartTimes] = time[nEnd]
                    ExpectedBumpStartTimes[Num_ExpectedBumpStartTimes] = time[nStart]
                    BumpPanicTimes[Num_ExpectedBumpStartTimes] = panicTimes[j]
                    # ExpectedBumpStartTimes[Num_ExpectedBumpStartTimes] = panicTimes[j]
                    Num_ExpectedBumpStartTimes = Num_ExpectedBumpStartTimes + 1
                    # Color ext video red.
                    BUMP = True

    TRUE_PANIC_FOUND = False
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
            TruePanicIndices[j:j + 4] = panicIndices[j:j + 4]
            TRUE_PANIC_FOUND = True
            # break # for now only show the first time panic is detected.

    return PANIC, BUMP, panicTrue[0:numPanics], panicTimes[0:numPanics], panicIndices[0:numPanics], panicSeverity[
                                                                                                    0:numPanics], \
           Num_ExpectedBumpStartTimes, ExpectedBumpStartTimes[0:Num_ExpectedBumpStartTimes], \
           ExpectedBumpEndTimes[0:Num_ExpectedBumpStartTimes], BumpPanicTimes[0:Num_ExpectedBumpStartTimes], \
           TRUE_PANIC_FOUND, TruePanicIndices[0:numPanics]


def DetectSpeedBump(time, accx, gpstime, gpsspeed):  # ,tit):

    # Determine if a speed bump was found. If so, assume that any brake severity
    # within 0.3s before and 2.0s after are speed bump related an not actual hare braking.  These timing
    # checks are performed in function FindPanicBrakingEvents.  In the current function, only bump detection
    # is performed.

    SBdataframe = []
    SBdf = []

    KIJK = False  # Don't plot anything if set to False.

    NumBumpEvents = 0

    # GPS at 1Hz, IMU at 200Hz

    # Filter accx w 10Hz lpf
    [b, a] = butter_lowpass(6, 200, 4)
    sig = sp.signal.filtfilt(b, a, accx)
    N = len(sig)

    peaks = np.zeros(N)  # Not likely to have that many but just in case of crazy noise.
    peaktimes = np.zeros(N)
    peakindices = np.zeros(N).astype(int)
    peakspeeds = np.zeros(N)  # Speed at which the peak occurs - needed to set
    # time spacing between peaks based on assumed 2m wheel base
    numpeaks = 0

    # Find peaks in sig
    for n in np.arange(1, N - 1):
        # Pos peak
        if ((sig[n] > sig[n - 1]) & (sig[n] > sig[n + 1])):
            peaks[numpeaks] = sig[n]
            peaktimes[numpeaks] = time[n]
            peakindices[numpeaks] = n
            # Find nearest time in gps and set peak speed.
            i = np.argmin(np.absolute(gpstime - time[n]))
            peakspeeds[numpeaks] = gpsspeed[i]
            numpeaks = numpeaks + 1

        # Neg peak
        if ((sig[n] < sig[n - 1]) & (sig[n] < sig[n + 1])):
            peaks[numpeaks] = sig[n]
            peaktimes[numpeaks] = time[n]
            peakindices[numpeaks] = n
            # Find nearest time in gps and set peak speed.
            i = np.argmin(np.absolute(gpstime - time[n]))
            peakspeeds[numpeaks] = gpsspeed[i]
            numpeaks = numpeaks + 1

    # Detect if a positive and negative accx bump are of similar height and within wheel axis / speed seconds
    wheelbase = 2.0  # Assumed for all vehicle for now.
    DETECTED_BUMP = False
    bumpindices = np.zeros(N).astype(int)
    bumpstartindices = np.zeros(N).astype(int)
    bumpendindices = np.zeros(N).astype(int)
    bumpdieindices = np.zeros(N).astype(int)
    bumpseverities = np.zeros(N)
    bumpconfidences = np.zeros(N)
    numbumps = 0
    for k in np.arange(3, numpeaks - 1):
        # This algo focusses on speed bumps but may also capture a number of dips.    A dip is the inverse of a bump.
        # If th ebumps work well, the same algo should be inverted in peak signs to make an equally good dip detector.
        # Diagonal traversal through a dip or over a bump has a strong lateral accel component for which a separate
        # detector will be developed.
        # if ( (peaks[k-1] < -2.0) & (peaks[k] > 2.0) & ((peaktimes[k] - peaktimes[k-1]) < (wheelbase/peakspeeds[k-1] + 0.1)) ):
        # if ( (peaks[k-1] < -1.5) & (peaks[k] > 1.0) & ((peaktimes[k] - peaktimes[k-1]) < (wheelbase/peakspeeds[k-1] + 0.3)) ):

        if ((peaks[k - 1] < -1.5) and (peaks[k] > 1.0) and
                (peakspeeds[k - 1] > 0) and
                ((peaktimes[k] - peaktimes[k - 1]) < (wheelbase / peakspeeds[k - 1] + 0.3))):
            # We need to make sure that the peak before the 1st peak is at least twice as small in magnitude.
            # Otherwise noisy would constantly trigger bumps.
            # Also make sure that the previous peak is of opposite sign because otherwise it misses cases where the
            # driver may have been braking prior to hitting the bump.
            if (np.abs(peaks[k - 2]) > 0.5 * np.abs(peaks[k - 1])) & (peaks[k - 1] * peaks[k - 2] < 0.0):
                continue
            # The two peaks also need to be of similar magnitude.
            # This is only an issue when the 2nd peak (positive peak) is more than twice at large as the first.
            # if (  ((np.abs(peaks[k-1]) / np.abs(peaks[k])) > 2.0) | (((np.abs(peaks[k]) / np.abs(peaks[k-1])) > 2.0)) ):
            if (((np.abs(peaks[k]) / np.abs(peaks[k - 1])) > 2.0)):
                continue
            # Make sure that the speed is not nearly zero at the 2nd peak.
            # We need to avoid the situation where the car comes to a stand still and bounces in its suspension.
            # Check if the speed 2s later is zero.  If so, then it is not a bump.  Of course possible that
            # brake on a bump nearly to zero speed but that is quite unlikely.
            n_check = peakindices[k] + 200  # 1s later to allow for suspension to settle.
            if (n_check < len(time)):
                i = np.argmin(
                    np.absolute(gpstime - time[n_check]))  # Find nearest GPS index at possible speed bump time.
                if (gpsspeed[
                    i] > 0.5):  # At a very slow speed, a strong bump is not possible so accel profile must have originated differently.
                    bumpindices[numbumps] = peakindices[k - 1]
                    bumpstartindices[numbumps] = peakindices[k - 2]
                    bumpendindices[numbumps] = peakindices[k + 1]
                    bumpdieindices[numbumps] = peakindices[k + 1] + 3 * (
                                peakindices[k + 1] - peakindices[k - 2])  # Die out effect when
                    # 2nd or other wheels also go over bump. These are harder to detect because of strong wheel
                    # base and suspension effects and speed chnage effects.
                    # Hence it is estimatewd as an additional twice as long as front hweel bump period.
                    # Bump severity is defined as the mean of the pos and neg peak.
                    bumpseverities[numbumps] = 0.5 * (np.abs(peaks[k]) + np.abs(peaks[k - 1]))
                    # Confidence is the minim of ratio A/B,B/A.  if close to 1 then perfect energy balance.  Can also integrate under curves
                    # but issue is that does not always start at zero.  In fact of breake before hittong bump, then speed bump conf slightly
                    # less than 1.
                    bumpconfidences[numbumps] = np.min(
                        [np.abs(peaks[k]) / np.abs(peaks[k - 1]), np.abs(peaks[k - 1]) / np.abs(peaks[k])])
                    DETECTED_BUMP = True

                    # Now find the start and end of the bump which is from peak k-2 to peak k+1

                    SBdataframe = pd.DataFrame({'DETECTED_BUMP': [DETECTED_BUMP],
                                                'NumBumps': [numbumps],
                                                'BumpStartIndices': [bumpstartindices[numbumps]],
                                                'BumpEndIndices': [bumpendindices[numbumps]],
                                                'BumpDieIndices': [bumpdieindices[numbumps]],
                                                'BumpIndices': [bumpindices[numbumps]],
                                                # indices within input arrays where bump occurs
                                                'BumpSeverities': [bumpseverities[numbumps]],
                                                # mean of two peaks that define the bump.
                                                'BumpConfidences': [bumpconfidences[numbumps]],
                                                # minimum ratio of the two peaks [A/B, B/A]
                                                'device_id': -1,
                                                # Add these to that they can be filled in from calling function.
                                                'event_id': -1,
                                                'file_index': -1,
                                                'time': -1})

                    numbumps = numbumps + 1

                    if (NumBumpEvents == 0):
                        SBdf = SBdataframe
                    else:
                        SBdf = SBdf.append(SBdataframe, ignore_index=True)
                    NumBumpEvents = NumBumpEvents + 1

    # Plot raw data.
    if (KIJK):
        fig = plt.figure(333)
        plt.clf()
        ax = fig.add_subplot(111)
        ax.plot(time, accx, ':m')
        ax.plot(time, sig, 'r')
        ax.plot(gpstime, gpsspeed, 'b')
        ax.grid()
        # ax.set_title(tit)

        # Show peaks
        ax.plot(peaktimes[0:numpeaks], peaks[0:numpeaks], 'kd', MarkerFaceColor='y')

        # Show bumps.
        if (DETECTED_BUMP):
            ax.plot(time[bumpindices[0:numbumps]], sig[bumpindices[0:numbumps]], 'kd', MarkerFaceColor='m')
            for j in range(numbumps):
                t = time[bumpindices[j]]
                ax.plot([t, t], [-5, 5], 'g', LineWidth=2)

        plt.show()
        plt.pause(1.0)

    # Boolean if any bumps are detected.  If so, bump indices holds indices in original time and accx
    # where the bumps was detected i.e. location of first negative peak.

    return SBdf


def EstimVehTrajRadius(time, accy1000, gyrz1000, speed):
    # Speed is the GPS speed interpolated to the IMU speed.  This is done in the
    # utility function N2_data_ingest.py imported above
    # The gps variables are at original 1Hz.

    # Three different radii estimates are provided.
    # 1) purely from IMU --> Rimu
    # 2) imu and gps speed --> Rimugpsspeed
    # 3) gps locations only --> Rgpslonlat

    # Each of them returns the radius at the IMU resolution.

    N = len(time)
    Rimu = np.zeros(N)
    Rimugpsspeed = np.zeros(N)
    Rgpslonlat = np.zeros(N)

    # Alternative estimate of radius when GPS speed available.  This is quite stable.
    radiusGPS = np.zeros(N)
    vEst = np.zeros(N)
    radiusEst = np.zeros(N)

    # Compute amount turned by looking when abs(gz1000) > 0.1 - start integration until below 0.1 again.
    # Then walk back from start and forward from end.
    turnAng = np.zeros(N)
    Ts = 1.0 / 200.0
    ang = 0.0
    n = 1

    startCurve = 0  # verify !

    while (n < N):
        #         if (N != len(time)):
        #             print('N change: ',n,N, len(time))
        #         if (n>=N) or (n>=len(gyrz1000)):
        #             print('warning: ',n,N, len(time))
        if ((np.absolute(gyrz1000[n]) >= 0.1) & (np.absolute(gyrz1000[n - 1]) < 0.1)):
            startCurve = n
            ang = 0.0  # Reset
        if ((np.absolute(gyrz1000[n]) < 0.1) & (np.absolute(gyrz1000[n - 1]) >= 0.1)):
            ang = 0.0  # Reset
            endCurve = n

            peakCurve = np.floor(0.5 * (startCurve + endCurve)).astype(int)

            # Now we can walk forward and backwards to find the true start and end of curve and recompute the turn angle.

            nn = startCurve
            # Check if a peak is reached - does not matter if pos or neg.

            # Walk back until sign changes
            while ((np.sign(gyrz1000[nn]) == np.sign(gyrz1000[peakCurve]))):
                nn = nn - 1
                if (nn < 1):
                    break
            startCurveTrue = nn

            nn = endCurve
            # Check if a peak is reached - does not matter if pos or neg.
            # Walk forward until sign changes
            while ((np.sign(gyrz1000[nn]) == np.sign(gyrz1000[peakCurve]))):
                nn = nn + 1
                if (nn > N - 2):
                    break
            endCurveTrue = nn

            # Integrate yawrate over this whole interval.
            ta = 0.0  # Turn angle
            for nn in range(startCurveTrue, endCurveTrue):
                ta = ta + Ts * gyrz1000[nn]  # gz[nn]
            turnAng[startCurveTrue:endCurveTrue] = ta

        # To compute radius estimate for 0.5s back simply use the following:
        if ((np.absolute(gyrz1000[n]) > 0.1) & (np.absolute(accy1000[n]) > 0.5)):
            vEst[n] = accy1000[n] / gyrz1000[n]
            radiusEst[n] = vEst[n] / gyrz1000[n]  # Note that this is from 0.5s ago.
            radiusGPS[n] = speed[n] / gyrz1000[n]

        n = n + 1

    TurnAng = turnAng
    Rimu = radiusEst
    Vimu = vEst
    Rimugpsspeed = radiusGPS

    return TurnAng, Rimu, Vimu, Rimugpsspeed
