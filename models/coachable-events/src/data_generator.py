import h5py
import math
import random
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from scipy.signal import butter
from tensorflow.keras.utils import Sequence
# from keras.utils.np_utils import to_categorical


def butter_lowpass(cutoff: float, fs: float, order: int = 5):
    f_nyq = 0.5 * fs
    cof_normalized = cutoff / f_nyq
    b, a = butter(N=order, Wn=cof_normalized, btype='low', analog=False, output='ba')
    return b, a


class DataGenerator_Conv1D_v2(Sequence):
    """
    Modified from above DataGenerator_Conv1D.
    Key difference is simply the shape being transposed from (batch_size, n_features, window_len) to
    (batch_size, window_len, n_features)
    so better suited for Conv1D models
    DONE: add the handling of input channels
    DONE: add the cutting around peak_ts
    DONE: add the support for down-sampling/up-sampling to desired frequency
    DONE: change the generate_y to generate multi-class vector
    DONE: add the support of stream normalization/standardization
    DONE: add other low-pass filtering preprocessing to IMU channels before interpolation
    DONE: refactor/modularize the get_X_from_ith_sample function
    TODO: will be good to run some data analysis to find good normalization factors
    """

    def __init__(self, **kwargs):
        # dataset
        self.events = kwargs.get('events')

        # h5 files
        self.h5_files = self.events.dbfs_h5_file.values
        print('Initializing data generator... %d files found' % len(self.h5_files))

        # labels
        self.labels = kwargs.get('labels')
        self.n_class = kwargs.get('n_class')
        # check the dimension match
        if self.labels.shape[1] != self.n_class:
            raise ValueError(f'Dimension mismatch: labels.shape[1]={self.labels.shape[1]} while n_class={self.n_class}')

        self.batch_size = kwargs.get('batch_size', 1)
        self.n_samples = self.get_num_samples()
        print('Total samples: %d' % self.n_samples)
        self.indices = list(range(self.n_samples))

        # channels
        self.available_channels = self.get_available_stream_names_from_h5()
        input_channels = kwargs.get('channels')
        for channel in input_channels:
            if channel not in self.available_channels:
                print(f'input channel: {channel} in unavailable from the h5 files, removed.')
        self.channels = [channel for channel in input_channels if channel in self.available_channels]
        self.n_channels = len(self.channels)
        self.output_X_type = kwargs.get('output_X_type')

        # data trimming and padding
        self.event_peak_ts = self.events.event_peak_ts_unix.values
        self.n_sec_before_peak = kwargs.get('n_sec_before_peak')
        self.n_sec_after_peak = kwargs.get('n_sec_after_peak')
        self.pad_value = -1

        # data preprocess-filtering
        self.filter_params = kwargs.get('preprocess_filter_params', None)
        if self.filter_params is not None:
            self.is_filter = True
        else:
            self.is_filter = False

        # data down-sampling and up-sampling
        self.sample_rate = kwargs.get('sample_rate')
        self.window_len = int((self.n_sec_before_peak + self.n_sec_after_peak)*self.sample_rate) + 1
        print(f'Desired sample_rate = {self.sample_rate} Hz. '
              f'Time window = {self.n_sec_before_peak + self.n_sec_after_peak} seconds. '
              f'Calculated window_len = {self.window_len}.')

        # data normalization or standardization
        self.normalization_factors = kwargs.get('normalization_factors', None)
        if self.normalization_factors is not None:
            self.is_normalize = True
        else:
            self.is_normalize = False

        # print debug information
        self.is_debug = kwargs.get('is_debug')

        if kwargs.get('shuffle_init', False) is True:
            self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(self.get_num_samples() / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        start_ind = index * self.batch_size
        end_ind = (index + 1) * self.batch_size
        indices_for_current_batch = self.indices[start_ind:end_ind]

        # Generate data
        X = self._generate_X(indices_for_current_batch)
        y = self._generate_y(indices_for_current_batch)

        return X, y

    def get_num_samples(self):
        # for now, we will return 1 sample per file
        return len(self.h5_files)

    def get_available_stream_names_from_h5(self):
        h5_filename = self.h5_files[0]
        with h5py.File(h5_filename, 'r') as hf:
            return list(hf.keys())

    def on_epoch_end(self):
        # Updates indexes after each epoch
        random.shuffle(self.indices)

    def _apply_filter(self, data_in, param_key):
        # create filter
        if self.filter_params[param_key]['filter_type'] == 'butter_lowpass':
            b, a = butter_lowpass(cutoff=self.filter_params[param_key]['filter_cof'],
                                  fs=200,
                                  order=self.filter_params[param_key]['filter_order'])
        else:
            raise ValueError(f"The specified filter_type '{self.filter_params[param_key]['filter_type']}' is "
                             f"not supported.")

        # apply filter
        if self.filter_params[param_key]['filter_apply_func'] == 'filtfilt':
            data_out = sp.signal.filtfilt(b, a, data_in, axis=0)
            return data_out
        else:
            raise ValueError(f"The specified filter_apply_func '{self.filtering[param_key]['filter_apply_func']}' is "
                             f"not supported.")

    def _normalize_data(self, data_in, param_key: str):
        return data_in / self.normalization_factors[param_key]

    def _interpolate_data(self, t_in, data_in, t_out):
        if t_in.shape[0] < 2:
            if self.is_debug is True:
                print('DEBUG: data_raw has less than 2 points; unable to interpolate. Simply return zero array.')
            data_out = np.zeros((t_out.shape[0], data_in.shape[1]))
        else:
            if self.is_debug is True:
                print('DEBUG: interpolation time range: data_raw[:,0]: ({0}, {1}); t_out: ({2}, {3})'.
                      format(t_in.min().round(3), t_in.max().round(3), t_out.min().round(3), t_out.max().round(3)))
            interpolator = interp1d(t_in, data_in, axis=0, bounds_error=False, fill_value=self.pad_value, kind='linear')
            data_out = interpolator(t_out).reshape((t_out.shape[0], data_in.shape[1]))
        return data_out

    def get_X_from_ith_sample(self, file_idx: int, output_type: str = 'tuple'):

        if output_type not in ('matrix', 'tuple'):
            raise ValueError(f'parameter output_type: "{output_type}" is not supported.')

        h5_filename = self.h5_files[file_idx]
        event_peak_ts = self.event_peak_ts[file_idx]
        dt = 1 / self.sample_rate

        data_out = []

        # load data from h5
        hf = h5py.File(h5_filename, 'r')

        # determine the timestamps to interpolate to
        # use event_peak_ts as the center, and trim to target before and after
        t_out = np.arange(event_peak_ts-self.n_sec_before_peak, event_peak_ts+self.n_sec_after_peak+dt, dt)

        for ind, channel in enumerate(self.channels):
            data_raw = hf.get(channel)
            t_in = data_raw[:, 0] / 1e9

            if self.is_debug is True:
                print(f'DEBUG: Adding channel {ind}: {channel}')
                print(f'DEBUG: data_raw.shape = {data_raw.shape}')

            # specify which channels to extract
            if channel in ('gps_data', ):
                stream_idx_to_use = np.array([4])  # for gps, use only the speed stream
            elif channel in ('dist_multilabel_data', ):
                stream_idx_to_use = np.arange(1, 10)  # for dist_multilabel, use all streams
            elif channel in ('tailgating_data', ):
                stream_idx_to_use = np.array([2])  # for tg, use only the distance_estimate stream
            elif channel in ('fcw_data', ):
                stream_idx_to_use = np.array([1])  # for fcw, use only the ttc stream
            else:
                stream_idx_to_use = np.arange(1, 4)  # for oriented_acc and oriented_gyro, use all 3 streams (x,y,z)
            data_to_use = data_raw[:, stream_idx_to_use]

            # apply filter
            if self.is_filter is True:
                if channel in self.filter_params:
                    data_to_use = self._apply_filter(data_in=data_to_use, param_key=channel)

            # apply normalization
            if self.is_normalize is True:
                if channel in self.normalization_factors:
                    data_to_use = self._normalize_data(data_in=data_to_use, param_key=channel)

            # apply interpolation
            if self.is_debug is True:
                print(f'DEBUG: event_peak_ts: {event_peak_ts.round(3)}')
            data_to_use = self._interpolate_data(t_in=t_in, data_in=data_to_use, t_out=t_out)
            if self.is_debug is True:
                print(f'DEBUG: data_to_use.shape = {data_to_use.shape}')

            # append to data_out
            if output_type == 'matrix':  # return as matrix
                if ind == 0:
                    # initialize data_out
                    data_out = data_to_use
                else:
                    data_out = np.hstack((data_out, data_to_use))
                if self.is_debug is True:
                    print(f'DEBUG: data_out.shape = {data_out.shape}')

            elif output_type == 'tuple':  # return as tuple
                data_out.append(data_to_use)

        hf.close()
        if output_type == 'tuple':
            # convert from list of arrays to tuple of arrays
            data_out = tuple(data_out)

        return data_out

    def get_unshuffled_labels(self):
        return self.labels

    def _generate_X(self, indexes):
        """
        Generates data containing batch_size samples
        depends on the parameter output_X_type:
            if 'matrix': X = (batch_size, window_len, 18)
            if 'tuple':  X = (X_ACC, X_GYR, X_SPEED, X_DIST, X_DISTANCE, X_TTC)
        """
        # initialize
        if self.output_X_type == 'matrix':
            X = np.empty((self.batch_size, self.window_len, 18))

            for i, idx in enumerate(indexes):
                data = self.get_X_from_ith_sample(file_idx=idx, output_type=self.output_X_type)

                X[i, ] = data[:self.window_len, :]

            return X

        elif self.output_X_type == 'tuple':
            X_ACC = np.empty((self.batch_size, self.window_len, 3))
            X_GYRO = np.empty((self.batch_size, self.window_len, 3))
            X_SPEED = np.empty((self.batch_size, self.window_len, 1))
            X_DIST = np.empty((self.batch_size, self.window_len, 9))
            X_DISTANCE = np.empty((self.batch_size, self.window_len, 1))
            X_TTC = np.empty((self.batch_size, self.window_len, 1))

            for i, idx in enumerate(indexes):
                data_acc, data_gyro, data_speed, data_dist, data_distance, data_ttc = \
                    self.get_X_from_ith_sample(file_idx=idx, output_type=self.output_X_type)

                X_ACC[i, ] = data_acc[:self.window_len, :]
                X_GYRO[i, ] = data_gyro[:self.window_len, :]
                X_SPEED[i, ] = data_speed[:self.window_len, :]
                X_DIST[i, ] = data_dist[:self.window_len, :]
                X_DISTANCE[i, ] = data_distance[:self.window_len, :]
                X_TTC[i, ] = data_ttc[:self.window_len, :]

            return X_ACC, X_GYRO, X_SPEED, X_DIST, X_DISTANCE, X_TTC
        else:
            raise ValueError(f'parameter output_X_type: "{self.output_X_type}" is not supported.')

    def _generate_y(self, indexes):
        """
        Generates data containing batch_size masks
        """
        y = np.empty((self.batch_size, self.n_class))

        for i, idx in enumerate(indexes):
            y[i, ] = self.labels[idx]

        return y


class DataGenerator_SlidingWindow(Sequence):
    """
    For each event h5, this data generator creates N sliding windows with the shape of (N, window_len, n_features),
        so we can create a time-series evaluation of events.
    20220603_1730: can load data now,
    TODO: need to modify the time_window idea, no longer need to use n_sec before peak and after peak
    """

    def __init__(self, **kwargs):
        # dataset
        self.events = kwargs.get('events')

        # h5 files
        self.h5_files = self.events.dbfs_h5_file.values
        self.n_file = len(self.h5_files)
        print(f'Initializing data generator... {self.n_file} files found')
        self.indices = list(range(self.n_file))

        # labels
        self.labels = kwargs.get('labels')
        self.n_class = kwargs.get('n_class')
        # check the dimension match
        if self.labels.shape[1] != self.n_class:
            raise ValueError(f'Dimension mismatch: labels.shape[1]={self.labels.shape[1]} while n_class={self.n_class}')

        # channels
        self.available_channels = self.get_available_stream_names_from_h5()
        input_channels = kwargs.get('channels')
        for channel in input_channels:
            if channel not in self.available_channels:
                print(f'input channel: {channel} in unavailable from the h5 files, removed.')
        self.channels = [channel for channel in input_channels if channel in self.available_channels]
        self.n_channel = len(self.channels)
        self.output_X_type = kwargs.get('output_X_type', 'tuple')
        if self.output_X_type not in ('matrix', 'tuple'):
            raise ValueError(f'parameter output_X_type: "{self.output_X_type}" is not supported.')

        # data trimming and padding
        self.event_peak_ts = self.events.event_peak_ts_unix.values
        self.n_sec_before_peak = kwargs.get('n_sec_before_peak')
        self.n_sec_after_peak = kwargs.get('n_sec_after_peak')
        self.n_sec_total = self.n_sec_before_peak + self.n_sec_after_peak
        self.pad_value = -1

        # data preprocess-filtering
        self.filter_params = kwargs.get('preprocess_filter_params', None)
        if self.filter_params is not None:
            self.is_filter = True
        else:
            self.is_filter = False

        # data down-sampling and up-sampling parameters
        self.sample_rate = kwargs.get('sample_rate')
        self.dt = 1 / self.sample_rate
        self.window_len = int(self.n_sec_total * self.sample_rate) + 1
        print(f'Desired sample_rate = {self.sample_rate} Hz. '
              f'Time window = {self.n_sec_total} seconds. '
              f'Calculated window_len = {self.window_len}.')

        # data normalization or standardization
        self.normalization_factors = kwargs.get('normalization_factors', None)
        if self.normalization_factors is not None:
            self.is_normalize = True
        else:
            self.is_normalize = False

        # get batch_size and key_timestamps from each
        print('In the "SlidingWindow mode, each batch returns N sliding windows from event[idx], '
              'where N depends on the event itself.')
        self.batch_size = 1
        self.key_timestamps = self.get_key_timestamps()
        self.n_sample_per_file = self.get_num_samples_per_file()
        self.n_sample = self.get_num_samples()
        print('Total samples: %d' % self.n_sample)

        # print debug information
        self.is_debug = kwargs.get('is_debug')

        if kwargs.get('shuffle_init', False) is True:
            self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        # return int(np.floor(self.get_num_samples() / self.batch_size))
        return int(np.floor(self.n_file / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        start_ind = index * self.batch_size
        end_ind = (index + 1) * self.batch_size
        indices_for_current_batch = self.indices[start_ind:end_ind]

        # Generate data
        X = self._generate_X(indices_for_current_batch)
        y = self._generate_y(indices_for_current_batch)

        return X, y

    def get_key_timestamps(self):
        # In this SlidingWindow mode, the data generator will return N time windows for each event as one batch.
        # i.e. batch_size = N
        # where N is determined by the data length per event, and the sampling_rate specified
        # this function calculate the key timestamps and the batch_sizes for each event
        key_timestamps = []

        for file_idx in self.indices:
            h5_filename = self.h5_files[file_idx]
            event_peak_ts = self.event_peak_ts[file_idx]

            # load data from h5 and use the timestamps from ori_acc as the reference
            # to create the list of key timestamps
            hf = h5py.File(h5_filename, 'r')
            t_in = hf.get('ori_acc_data')[:, 0] / 1e9
            n_sample_before_peak = math.floor((event_peak_ts - t_in[0]) * self.sample_rate)
            n_sample_after_peak = math.floor((t_in[-1] - event_peak_ts) * self.sample_rate)
            key_ts = np.arange(event_peak_ts - n_sample_before_peak * self.dt,
                               event_peak_ts + (n_sample_after_peak + 1) * self.dt, self.dt)

            key_timestamps.append(key_ts)

        return key_timestamps

    def get_num_samples_per_file(self):
        if not self.key_timestamps:
            self.key_timestamps = self.get_key_timestamps()
        return [len(key_ts) for key_ts in self.key_timestamps]

    def get_num_samples(self):
        return sum(self.n_sample_per_file)

    def get_available_stream_names_from_h5(self):
        h5_filename = self.h5_files[0]
        with h5py.File(h5_filename, 'r') as hf:
            return list(hf.keys())

    def on_epoch_end(self):
        # Updates indexes after each epoch
        random.shuffle(self.indices)

    def _apply_filter(self, data_in, param_key):
        # create filter
        if self.filter_params[param_key]['filter_type'] == 'butter_lowpass':
            b, a = butter_lowpass(cutoff=self.filter_params[param_key]['filter_cof'],
                                  fs=200,
                                  order=self.filter_params[param_key]['filter_order'])
        else:
            raise ValueError(f"The specified filter_type '{self.filter_params[param_key]['filter_type']}' is "
                             f"not supported.")

        # apply filter
        if self.filter_params[param_key]['filter_apply_func'] == 'filtfilt':
            data_out = sp.signal.filtfilt(b, a, data_in, axis=0)
            return data_out
        else:
            raise ValueError(f"The specified filter_apply_func '{self.filtering[param_key]['filter_apply_func']}' is "
                             f"not supported.")

    def _normalize_data(self, data_in, param_key: str):
        return data_in / self.normalization_factors[param_key]

    def _interpolate_data(self, t_in, data_in, t_out):
        if t_in.shape[0] < 2:
            if self.is_debug is True:
                print('DEBUG: data_raw has less than 2 points; unable to interpolate. Simply return zero array.')
            data_out = np.zeros((t_out.shape[0], data_in.shape[1]))
        else:
            if self.is_debug is True:
                print('DEBUG: interpolation time range: data_raw[:,0]: ({0}, {1}); t_out: ({2}, {3})'.
                      format(t_in.min().round(3), t_in.max().round(3), t_out.min().round(3), t_out.max().round(3)))
            interpolator = interp1d(t_in, data_in, axis=0, bounds_error=False, fill_value=self.pad_value, kind='linear')
            data_out = interpolator(t_out).reshape((t_out.shape[0], data_in.shape[1]))
        return data_out

    def get_X_from_ith_file(self, file_idx: int):

        h5_filename = self.h5_files[file_idx]
        event_peak_ts = self.event_peak_ts[file_idx]
        key_timestamps = self.key_timestamps[file_idx]
        batch_size = self.n_sample_per_file[file_idx]

        # initialize the output data
        if self.output_X_type == 'matrix':
            data_out = np.empty((batch_size, self.window_len, 18))
        elif self.output_X_type == 'tuple':
            X_ACC = np.empty((batch_size, self.window_len, 3))
            X_GYRO = np.empty((batch_size, self.window_len, 3))
            X_SPEED = np.empty((batch_size, self.window_len, 1))
            X_DIST = np.empty((batch_size, self.window_len, 9))
            X_DISTANCE = np.empty((batch_size, self.window_len, 1))
            X_TTC = np.empty((batch_size, self.window_len, 1))

        hf = h5py.File(h5_filename, 'r')

        # start to iterate through the list of key timestamps and generate data
        for ts_idx, key_ts in enumerate(key_timestamps):
            data_tmp = []
            # determine the timestamps to interpolate to
            # use the key timestamp and trim/pad to the target seconds before and after
            t_out = np.arange(key_ts-self.n_sec_total, key_ts+self.dt, self.dt)

            for channel_idx, channel in enumerate(self.channels):
                data_raw = hf.get(channel)
                t_in = data_raw[:, 0] / 1e9

                if self.is_debug is True:
                    print(f'DEBUG: Adding channel {channel_idx}: {channel}')
                    print(f'DEBUG: data_raw.shape = {data_raw.shape}')

                # specify which channels to extract
                if channel in ('gps_data', ):
                    stream_to_extract_idx = np.array([4])  # for gps, use only the speed stream
                elif channel in ('dist_multilabel_data', ):
                    stream_to_extract_idx = np.arange(1, 10)  # for dist_multilabel, use all streams
                elif channel in ('tailgating_data', ):
                    stream_to_extract_idx = np.array([2])  # for tg, use only the distance_estimate stream
                elif channel in ('fcw_data', ):
                    stream_to_extract_idx = np.array([1])  # for fcw, use only the ttc stream
                else:
                    stream_to_extract_idx = np.arange(1, 4)  # for oriented_acc and oriented_gyro, use all 3 streams (x,y,z)
                data_extracted = data_raw[:, stream_to_extract_idx]

                # apply filter
                if self.is_filter is True:
                    if channel in self.filter_params:
                        data_extracted = self._apply_filter(data_in=data_extracted, param_key=channel)

                # apply normalization
                if self.is_normalize is True:
                    if channel in self.normalization_factors:
                        data_extracted = self._normalize_data(data_in=data_extracted, param_key=channel)

                # apply interpolation
                if self.is_debug is True:
                    print(f'DEBUG: event_peak_ts: {event_peak_ts.round(3)}')
                data_extracted = self._interpolate_data(t_in=t_in, data_in=data_extracted, t_out=t_out)
                if self.is_debug is True:
                    print(f'DEBUG: data_extracted.shape = {data_extracted.shape}')

                # append to data_out
                if self.output_X_type == 'matrix':  # return as matrix
                    if channel_idx == 0:
                        # initialize data_out
                        data_tmp = data_extracted
                    else:
                        data_tmp = np.hstack((data_tmp, data_extracted))
                    if self.is_debug is True:
                        print(f'DEBUG: data_out.shape = {data_tmp.shape}')

                elif self.output_X_type == 'tuple':  # return as tuple
                    data_tmp.append(data_extracted)

            if self.output_X_type == 'matrix':
                data_out[ts_idx, ] = data_tmp
            elif self.output_X_type == 'tuple':
                X_ACC[ts_idx, ] = data_tmp[0]
                X_GYRO[ts_idx, ] = data_tmp[1]
                X_SPEED[ts_idx, ] = data_tmp[2]
                X_DIST[ts_idx, ] = data_tmp[3]
                X_DISTANCE[ts_idx, ] = data_tmp[4]
                X_TTC[ts_idx, ] = data_tmp[5]

        hf.close()

        if self.output_X_type == 'tuple':
            data_out = (X_ACC, X_GYRO, X_SPEED, X_DIST, X_DISTANCE, X_TTC)

        return data_out

    def _generate_X(self, indexes):
        """
        Generates data containing batch_size samples
        depends on the parameter output_X_type:
            if 'matrix': X = (batch_size, window_len, 18)
            if 'tuple':  X = (X_ACC, X_GYR, X_SPEED, X_DIST, X_DISTANCE, X_TTC)
        """
        if len(indexes) > 1:
            raise ValueError(f"""The slidingWindow mode only supports generating data from 1 sample at a time; 
                              got {len(indexes)} instead.""")

        return self.get_X_from_ith_file(file_idx=indexes[0])


    def _generate_y(self, indexes):
        """
        Generates data containing batch_size masks
        """
        if len(indexes) > 1:
            raise ValueError(f"""The slidingWindow mode only supports generating data from 1 sample at a time; 
                              got {len(indexes)} instead.""")

        file_idx = indexes[0]
        batch_size = self.n_sample_per_file[file_idx]

        y = np.ones((batch_size, self.n_class)) * self.labels[file_idx]
        return y


# # comment out the test section for now -- does not work yet
# # since we need to download test data to local in order for the test to work
# # alternatively, have been testing on databricks using data on the s3
# if __name__ == '__main__':
#     import pandas as pd
#     # TODO: add test X, y, y_ohe
#     X = pd.DataFrame()
#     y = pd.Series()
#     y_ohe = []
#
#     # Specify params
#     datagen_params = {'events': X,
#                       'labels': y_ohe,
#                       'n_class': y.shape[1],
#                       'channels': ['ori_acc_data', 'ori_gyro_data', 'gps_data', 'dist_multilabel_data',
#                                    'tailgating_data', 'fcw_data'],
#                       'normalization_factors': {'ori_acc_data': 10.0,
#                                                 'ori_gyro_data': 0.02,
#                                                 'gps_data': 40,
#                                                 'tailgating_data': 100,
#                                                 'fcw_data': 10,
#                                                 },
#                       'preprocess_filter_params': {'ori_acc_data': {'filter_type': 'butter_lowpass',
#                                                                     'filter_order': 2,
#                                                                     'filter_cof': 10,
#                                                                     'filter_apply_func': 'filtfilt'},
#                                                    'ori_gyro_data': {'filter_type': 'butter_lowpass',
#                                                                      'filter_order': 2,
#                                                                      'filter_cof': 10,
#                                                                      'filter_apply_func': 'filtfilt'},
#                                                    },
#                       'n_sec_before_peak': 5.0,
#                       'n_sec_after_peak': 3.0,
#                       'sample_rate': 5,
#                       'batch_size': 4,
#                       'shuffle_init': False,
#                       'is_debug': False,
#                       'output_X_type': 'tuple'
#                       }
#
#     # initialize the data generator
#     dg = DataGenerator_Conv1D_v2(**datagen_params)
#
#     # specify the test data index
#     test_batch_idx = 1
#     test_event_idx = 2
#
#     # test between output type "matrix" vs "tuple"
#     tmp_X1 = dg.get_X_from_ith_sample(file_idx=datagen_params['batch_size'] * test_batch_idx + test_event_idx,
#                                       output_type='matrix')
#     tmp_X2 = dg.get_X_from_ith_sample(file_idx=datagen_params['batch_size'] * test_batch_idx + test_event_idx,
#                                       output_type='tuple')
#     assert (tmp_X1 == np.hstack(tmp_X2)).all()
#
#     # test the __getitem__()
#     tmp_X, tmp_y = dg.__getitem__(index=test_batch_idx)
#     tmp_X = tuple(X[test_event_idx, :, :] for X in tmp_X)
#     assert (np.hstack(tmp_X) == np.hstack(tmp_X2)).all()
#     print('Assert the "X". Passed.')
#     assert (tmp_y[test_event_idx, :] == y_ohe[datagen_params['batch_size'] * test_batch_idx + test_event_idx]).all()
#     print('Assert the "y". Passed.')
