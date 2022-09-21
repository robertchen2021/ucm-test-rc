"""
This file contains some IMU-based algorithms
Primary author: Erwin Boer
"""
from typing import List, Dict

import numpy as np
import pandas as pd
import scipy as sp
from scipy.signal import butter


def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    [b, a] = butter(order, normalCutoff, btype='low', analog=False)
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
            if ((time[nn] - time[n]) > 0.25) & (
                    nn != nBest):  # Only look 250ms ahead unless the brake severity is still growing.
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


def FindSevereBrakingEvents(file_index, TimeOffset, dev_id, mes_id, time, accx, gpsspeed):
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
            while (nn < Nbs - 1) and (decel[nn + 1] < decel[nn]):
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

    # # Plot raw data.
    # if (KIJK):
    #     fig = plt.figure(333)
    #     plt.clf()
    #     ax = fig.add_subplot(111)
    #     ax.plot(time, accx, ':m')
    #     ax.plot(time, sig, 'r')
    #     ax.plot(gpstime, gpsspeed, 'b')
    #     ax.grid()
    #     # ax.set_title(tit)
    #
    #     # Show peaks
    #     ax.plot(peaktimes[0:numpeaks], peaks[0:numpeaks], 'kd', MarkerFaceColor='y')
    #
    #     # Show bumps.
    #     if (DETECTED_BUMP):
    #         ax.plot(time[bumpindices[0:numbumps]], sig[bumpindices[0:numbumps]], 'kd', MarkerFaceColor='m')
    #         for j in range(numbumps):
    #             t = time[bumpindices[j]]
    #             ax.plot([t, t], [-5, 5], 'g', LineWidth=2)
    #
    #     plt.show()
    #     plt.pause(1.0)

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
