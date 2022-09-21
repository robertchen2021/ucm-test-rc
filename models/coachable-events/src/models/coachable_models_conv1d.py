from tensorflow.keras.layers import Concatenate, Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
from tensorflow.keras.models import Model


def get_model_conv1d_v3_late_fusion(**kwargs):
    """
    Adopted from conv1d_v3.
    Same number of stages as conv1d_v3, but move the concat to the end, and all conv1d layers before the concat
    """
    # Get params
    n_class = kwargs.get('n_class')
    x_type = kwargs.get('output_X_type', 'tuple')
    n_sec_before_peak = kwargs.get('n_sec_before_peak')
    n_sec_after_peak = kwargs.get('n_sec_after_peak')
    sample_rate = kwargs.get('sample_rate')
    padding = kwargs.get('padding', 'valid')
    dilation_rate = kwargs.get('dilation_rate', 1)
    n_conv1d_layer1 = kwargs.get('n_conv1d_layer1')
    n_conv1d_layer2 = kwargs.get('n_conv1d_layer2')
    n_conv1d_layer3 = kwargs.get('n_conv1d_layer3')
    n_conv1d_layer4 = kwargs.get('n_conv1d_layer4')
    n_conv1d_layer5 = kwargs.get('n_conv1d_layer5')
    size_conv1d_layer1 = kwargs.get('size_conv1d_layer1')
    size_conv1d_layer2 = kwargs.get('size_conv1d_layer2')
    size_conv1d_layer3 = kwargs.get('size_conv1d_layer3')
    size_conv1d_layer4 = kwargs.get('size_conv1d_layer4')
    size_conv1d_layer5 = kwargs.get('size_conv1d_layer5')
    pool_size1 = kwargs.get('pool_size1')

    # Get the input shape
    window_len = int((n_sec_before_peak + n_sec_after_peak) * sample_rate) + 1
    imu_input_shape = (window_len, 3)
    dist_input_shape = (window_len, 9)
    other_input_shape = (window_len, 1)

    # Check the type of input data
    if x_type != 'tuple':
        raise ValueError(f'The specified input type "{x_type}" is not supported.')

    X_acc_input = Input(imu_input_shape, name='X_acc_input')
    X_gyro_input = Input(imu_input_shape, name='X_gyro_input')
    X_speed_input = Input(other_input_shape, name='X_speed_input')
    X_dist_input = Input(dist_input_shape, name='X_dist_input')
    X_distance_input = Input(other_input_shape, name='X_distance_input')
    X_ttc_input = Input(other_input_shape, name='X_ttc_input')

    # sub-model for X_acc
    conv1d = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(X_acc_input)
    conv1d = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout = Dropout(0.5)(conv1d)
    conv1d = Conv1D(filters=n_conv1d_layer3, kernel_size=size_conv1d_layer3,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(dropout)
    conv1d = Conv1D(filters=n_conv1d_layer4, kernel_size=size_conv1d_layer4,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout_acc = Dropout(0.5)(conv1d)

    # sub-model for X_gyro
    conv1d = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(X_gyro_input)
    conv1d = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout = Dropout(0.5)(conv1d)
    conv1d = Conv1D(filters=n_conv1d_layer3, kernel_size=size_conv1d_layer3,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(dropout)
    conv1d = Conv1D(filters=n_conv1d_layer4, kernel_size=size_conv1d_layer4,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout_gyro = Dropout(0.5)(conv1d)

    # sub-model for X_speed
    conv1d = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(X_speed_input)
    conv1d = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout = Dropout(0.5)(conv1d)
    conv1d = Conv1D(filters=n_conv1d_layer3, kernel_size=size_conv1d_layer3,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(dropout)
    conv1d = Conv1D(filters=n_conv1d_layer4, kernel_size=size_conv1d_layer4,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout_speed = Dropout(0.5)(conv1d)

    # sub-model for X_dist
    conv1d = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(X_dist_input)
    conv1d = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout = Dropout(0.5)(conv1d)
    conv1d = Conv1D(filters=n_conv1d_layer3, kernel_size=size_conv1d_layer3,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(dropout)
    conv1d = Conv1D(filters=n_conv1d_layer4, kernel_size=size_conv1d_layer4,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout_dist = Dropout(0.5)(conv1d)

    # sub-model for X_distance
    conv1d = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(X_distance_input)
    conv1d = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout = Dropout(0.5)(conv1d)
    conv1d = Conv1D(filters=n_conv1d_layer3, kernel_size=size_conv1d_layer3,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(dropout)
    conv1d = Conv1D(filters=n_conv1d_layer4, kernel_size=size_conv1d_layer4,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout_distance = Dropout(0.5)(conv1d)

    # sub-model for X_ttc
    conv1d = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(X_ttc_input)
    conv1d = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout = Dropout(0.5)(conv1d)
    conv1d = Conv1D(filters=n_conv1d_layer3, kernel_size=size_conv1d_layer3,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(dropout)
    conv1d = Conv1D(filters=n_conv1d_layer4, kernel_size=size_conv1d_layer4,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout_ttc = Dropout(0.5)(conv1d)

    # merge
    merged = Concatenate()([dropout_acc, dropout_gyro, dropout_speed, dropout_dist, dropout_distance, dropout_ttc])

    # combined model after merge
    conv1d = Conv1D(filters=n_conv1d_layer5, kernel_size=size_conv1d_layer5,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(merged)
    maxpool1d = MaxPooling1D(pool_size=pool_size1)(conv1d)
    flatten = Flatten()(maxpool1d)

    # dense layers
    merged_output = Dropout(0.4)(flatten)
    merged_output = Dense(units=512, activation='relu')(merged_output)
    merged_output = Dense(units=64, activation='relu')(merged_output)

    # output layer
    Y = Dense(n_class, activation='softmax', name='coachable_output')(merged_output)

    loss = {'coachable_output': 'categorical_crossentropy'}
    model = Model(inputs=[X_acc_input, X_gyro_input, X_speed_input, X_dist_input, X_distance_input, X_ttc_input],
                  outputs=[Y])

    return model, loss, 'adam'


def get_model_conv1d_v3_early_fusion(**kwargs):
    """
    Adopted from conv1d_v3.
    Same number of stages as conv1d_v3, but move the concats to the beginning, and all conv1d layers after concat
    """
    # Get params
    n_class = kwargs.get('n_class')
    x_type = kwargs.get('output_X_type', 'tuple')
    n_sec_before_peak = kwargs.get('n_sec_before_peak')
    n_sec_after_peak = kwargs.get('n_sec_after_peak')
    sample_rate = kwargs.get('sample_rate')
    padding = kwargs.get('padding', 'valid')
    dilation_rate = kwargs.get('dilation_rate', 1)
    n_conv1d_layer1 = kwargs.get('n_conv1d_layer1')
    n_conv1d_layer2 = kwargs.get('n_conv1d_layer2')
    n_conv1d_layer3 = kwargs.get('n_conv1d_layer3')
    n_conv1d_layer4 = kwargs.get('n_conv1d_layer4')
    n_conv1d_layer5 = kwargs.get('n_conv1d_layer5')
    size_conv1d_layer1 = kwargs.get('size_conv1d_layer1')
    size_conv1d_layer2 = kwargs.get('size_conv1d_layer2')
    size_conv1d_layer3 = kwargs.get('size_conv1d_layer3')
    size_conv1d_layer4 = kwargs.get('size_conv1d_layer4')
    size_conv1d_layer5 = kwargs.get('size_conv1d_layer5')
    pool_size1 = kwargs.get('pool_size1')

    # Get the input shape
    window_len = int((n_sec_before_peak + n_sec_after_peak) * sample_rate) + 1
    imu_input_shape = (window_len, 3)
    dist_input_shape = (window_len, 9)
    other_input_shape = (window_len, 1)

    # Check the type of input data
    if x_type != 'tuple':
        raise ValueError(f'The specified input type "{x_type}" is not supported.')

    X_acc_input = Input(imu_input_shape, name='X_acc_input')
    X_gyro_input = Input(imu_input_shape, name='X_gyro_input')
    X_speed_input = Input(other_input_shape, name='X_speed_input')
    X_dist_input = Input(dist_input_shape, name='X_dist_input')
    X_distance_input = Input(other_input_shape, name='X_distance_input')
    X_ttc_input = Input(other_input_shape, name='X_ttc_input')
    X_sensor_data_input = Concatenate()([X_acc_input, X_gyro_input, X_speed_input, X_dist_input, X_distance_input,
                                         X_ttc_input])

    # combined model after merge
    conv1d = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(X_sensor_data_input)
    conv1d = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout = Dropout(0.5)(conv1d)
    conv1d = Conv1D(filters=n_conv1d_layer3, kernel_size=size_conv1d_layer3,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(dropout)
    conv1d = Conv1D(filters=n_conv1d_layer4, kernel_size=size_conv1d_layer4,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    conv1d = Conv1D(filters=n_conv1d_layer5, kernel_size=size_conv1d_layer5,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout = Dropout(0.5)(conv1d)
    maxpool1d = MaxPooling1D(pool_size=pool_size1)(dropout)
    flatten = Flatten()(maxpool1d)

    # dense layers
    merged_output = Dropout(0.4)(flatten)
    merged_output = Dense(units=512, activation='relu')(merged_output)
    merged_output = Dense(units=64, activation='relu')(merged_output)

    # output layer
    Y = Dense(n_class, activation='softmax', name='coachable_output')(merged_output)

    loss = {'coachable_output': 'categorical_crossentropy'}
    model = Model(inputs=[X_acc_input, X_gyro_input, X_speed_input, X_dist_input, X_distance_input, X_ttc_input],
                  outputs=[Y])

    return model, loss, 'adam'


def get_model_conv1d_v3(**kwargs):
    """
    Start with a few conv1d layers for each head, merge, then add a few more conv1d layers after merge
    - Based on conv1d_v2
    - Remove the 3rd conv1d layer, maxPooling1d, and flatten
    - Repeat the model after merging heads
    - Add the pool_size of maxPooling1d as param
    """
    # Get params
    n_class = kwargs.get('n_class')
    x_type = kwargs.get('output_X_type', 'tuple')
    n_sec_before_peak = kwargs.get('n_sec_before_peak')
    n_sec_after_peak = kwargs.get('n_sec_after_peak')
    sample_rate = kwargs.get('sample_rate')
    padding = kwargs.get('padding', 'valid')
    dilation_rate = kwargs.get('dilation_rate', 1)
    n_conv1d_layer1 = kwargs.get('n_conv1d_layer1')
    n_conv1d_layer2 = kwargs.get('n_conv1d_layer2')
    n_conv1d_layer3 = kwargs.get('n_conv1d_layer3')
    n_conv1d_layer4 = kwargs.get('n_conv1d_layer4')
    n_conv1d_layer5 = kwargs.get('n_conv1d_layer5')
    size_conv1d_layer1 = kwargs.get('size_conv1d_layer1')
    size_conv1d_layer2 = kwargs.get('size_conv1d_layer2')
    size_conv1d_layer3 = kwargs.get('size_conv1d_layer3')
    size_conv1d_layer4 = kwargs.get('size_conv1d_layer4')
    size_conv1d_layer5 = kwargs.get('size_conv1d_layer5')
    pool_size1 = kwargs.get('pool_size1')

    # Get the input shape
    window_len = int((n_sec_before_peak + n_sec_after_peak) * sample_rate) + 1
    imu_input_shape = (window_len, 3)
    dist_input_shape = (window_len, 9)
    other_input_shape = (window_len, 1)

    # Check the type of input data
    if x_type != 'tuple':
        raise ValueError(f'The specified input type "{x_type}" is not supported.')

    X_acc_input = Input(imu_input_shape, name='X_acc_input')
    X_gyro_input = Input(imu_input_shape, name='X_gyro_input')
    X_speed_input = Input(other_input_shape, name='X_speed_input')
    X_dist_input = Input(dist_input_shape, name='X_dist_input')
    X_distance_input = Input(other_input_shape, name='X_distance_input')
    X_ttc_input = Input(other_input_shape, name='X_ttc_input')

    # sub-model for X_acc
    conv1d = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(X_acc_input)
    conv1d = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout_acc = Dropout(0.5)(conv1d)

    # sub-model for X_gyro
    conv1d = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(X_gyro_input)
    conv1d = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout_gyro = Dropout(0.5)(conv1d)

    # sub-model for X_speed
    conv1d = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(X_speed_input)
    conv1d = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout_speed = Dropout(0.5)(conv1d)

    # sub-model for X_dist
    conv1d = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(X_dist_input)
    conv1d = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout_dist = Dropout(0.5)(conv1d)

    # sub-model for X_distance
    conv1d = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(X_distance_input)
    conv1d = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout_distance = Dropout(0.5)(conv1d)

    # sub-model for X_ttc
    conv1d = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(X_ttc_input)
    conv1d = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout_ttc = Dropout(0.5)(conv1d)

    # merge
    merged = Concatenate()([dropout_acc, dropout_gyro, dropout_speed, dropout_dist, dropout_distance, dropout_ttc])

    # combined model after merge
    conv1d = Conv1D(filters=n_conv1d_layer3, kernel_size=size_conv1d_layer3,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(merged)
    conv1d = Conv1D(filters=n_conv1d_layer4, kernel_size=size_conv1d_layer4,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    conv1d = Conv1D(filters=n_conv1d_layer5, kernel_size=size_conv1d_layer5,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    dropout = Dropout(0.5)(conv1d)
    maxpool1d = MaxPooling1D(pool_size=pool_size1)(dropout)
    flatten = Flatten()(maxpool1d)

    # dense layers
    merged_output = Dropout(0.4)(flatten)
    merged_output = Dense(units=512, activation='relu')(merged_output)
    merged_output = Dense(units=64, activation='relu')(merged_output)

    # output layer
    Y = Dense(n_class, activation='softmax', name='coachable_output')(merged_output)

    loss = {'coachable_output': 'categorical_crossentropy'}
    model = Model(inputs=[X_acc_input, X_gyro_input, X_speed_input, X_dist_input, X_distance_input, X_ttc_input],
                  outputs=[Y])

    return model, loss, 'adam'


def get_model_conv1d_v2(**kwargs):
    """
    Reference: https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
    """
    # Get params
    n_class = kwargs.get('n_class')
    x_type = kwargs.get('output_X_type', 'tuple')
    n_sec_before_peak = kwargs.get('n_sec_before_peak')
    n_sec_after_peak = kwargs.get('n_sec_after_peak')
    sample_rate = kwargs.get('sample_rate')
    n_conv1d_layer1 = kwargs.get('n_conv1d_layer1')
    n_conv1d_layer2 = kwargs.get('n_conv1d_layer2')
    n_conv1d_layer3 = kwargs.get('n_conv1d_layer3')
    size_conv1d_layer1 = kwargs.get('size_conv1d_layer1')
    size_conv1d_layer2 = kwargs.get('size_conv1d_layer2')
    size_conv1d_layer3 = kwargs.get('size_conv1d_layer3')

    # Get the input shape
    window_len = int((n_sec_before_peak + n_sec_after_peak) * sample_rate) + 1
    imu_input_shape = (window_len, 3)
    dist_input_shape = (window_len, 9)
    other_input_shape = (window_len, 1)

    # Check the type of input data
    if x_type != 'tuple':
        raise ValueError(f'The specified input type "{x_type}" is not supported.')

    X_acc_input = Input(imu_input_shape, name='X_acc_input')
    X_gyro_input = Input(imu_input_shape, name='X_gyro_input')
    X_speed_input = Input(other_input_shape, name='X_speed_input')
    X_dist_input = Input(dist_input_shape, name='X_dist_input')
    X_distance_input = Input(other_input_shape, name='X_distance_input')
    X_ttc_input = Input(other_input_shape, name='X_ttc_input')

    # sub-model for X_acc
    conv1d_1 = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1, activation='relu')(X_acc_input)
    conv1d_2 = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2, activation='relu')(conv1d_1)
    conv1d_3 = Conv1D(filters=n_conv1d_layer3, kernel_size=size_conv1d_layer3, activation='relu')(conv1d_2)
    dropout = Dropout(0.5)(conv1d_3)
    maxpool1d_1 = MaxPooling1D(pool_size=4)(dropout)
    maxpool1d_2 = MaxPooling1D(pool_size=4)(maxpool1d_1)
    flatten_acc = Flatten()(maxpool1d_2)

    # sub-model for X_gyro
    conv1d_1 = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1, activation='relu')(X_gyro_input)
    conv1d_2 = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2, activation='relu')(conv1d_1)
    conv1d_3 = Conv1D(filters=n_conv1d_layer3, kernel_size=size_conv1d_layer3, activation='relu')(conv1d_2)
    dropout = Dropout(0.5)(conv1d_3)
    maxpool1d_1 = MaxPooling1D(pool_size=4)(dropout)
    maxpool1d_2 = MaxPooling1D(pool_size=4)(maxpool1d_1)
    flatten_gyro = Flatten()(maxpool1d_2)

    # sub-model for X_speed
    conv1d_1 = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1, activation='relu')(X_speed_input)
    conv1d_2 = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2, activation='relu')(conv1d_1)
    conv1d_3 = Conv1D(filters=n_conv1d_layer3, kernel_size=size_conv1d_layer3, activation='relu')(conv1d_2)
    dropout = Dropout(0.5)(conv1d_3)
    maxpool1d_1 = MaxPooling1D(pool_size=4)(dropout)
    maxpool1d_2 = MaxPooling1D(pool_size=4)(maxpool1d_1)
    flatten_speed = Flatten()(maxpool1d_2)

    # sub-model for X_dist
    conv1d_1 = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1, activation='relu')(X_dist_input)
    conv1d_2 = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2, activation='relu')(conv1d_1)
    conv1d_3 = Conv1D(filters=n_conv1d_layer3, kernel_size=size_conv1d_layer3, activation='relu')(conv1d_2)
    dropout = Dropout(0.5)(conv1d_3)
    maxpool1d_1 = MaxPooling1D(pool_size=4)(dropout)
    maxpool1d_2 = MaxPooling1D(pool_size=4)(maxpool1d_1)
    flatten_dist = Flatten()(maxpool1d_2)

    # sub-model for X_distance
    conv1d_1 = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1, activation='relu')(X_distance_input)
    conv1d_2 = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2, activation='relu')(conv1d_1)
    conv1d_3 = Conv1D(filters=n_conv1d_layer3, kernel_size=size_conv1d_layer3, activation='relu')(conv1d_2)
    dropout = Dropout(0.5)(conv1d_3)
    maxpool1d_1 = MaxPooling1D(pool_size=4)(dropout)
    maxpool1d_2 = MaxPooling1D(pool_size=4)(maxpool1d_1)
    flatten_distance = Flatten()(maxpool1d_2)

    # sub-model for X_ttc
    conv1d_1 = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1, activation='relu')(X_ttc_input)
    conv1d_2 = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2, activation='relu')(conv1d_1)
    conv1d_3 = Conv1D(filters=n_conv1d_layer3, kernel_size=size_conv1d_layer3, activation='relu')(conv1d_2)
    dropout = Dropout(0.5)(conv1d_3)
    maxpool1d_1 = MaxPooling1D(pool_size=4)(dropout)
    maxpool1d_2 = MaxPooling1D(pool_size=4)(maxpool1d_1)
    flatten_ttc = Flatten()(maxpool1d_2)

    # merge
    merged_output = Concatenate()(
        [flatten_acc, flatten_gyro, flatten_speed, flatten_dist, flatten_distance, flatten_ttc])
    merged_output = Dropout(0.4)(merged_output)
    merged_output = Dense(units=512, activation='relu')(merged_output)
    merged_output = Dense(units=64, activation='relu')(merged_output)

    # output layer
    Y = Dense(n_class, activation='softmax', name='coachable_output')(merged_output)

    loss = {'coachable_output': 'categorical_crossentropy'}
    model = Model(inputs=[X_acc_input, X_gyro_input, X_speed_input, X_dist_input, X_distance_input, X_ttc_input],
                  outputs=[Y])

    return model, loss, 'adam'


def get_model_conv1d_simple(**kwargs):
    # Get the model params
    n_class = kwargs.get('n_class')
    x_type = kwargs.get('output_X_type', 'tuple')
    n_sec_before_peak = kwargs.get('n_sec_before_peak')
    n_sec_after_peak = kwargs.get('n_sec_after_peak')
    sample_rate = kwargs.get('sample_rate')
    padding = kwargs.get('padding', 'valid')
    dilation_rate = kwargs.get('dilation_rate', 1)
    n_conv1d_layer1 = kwargs.get('n_conv1d_layer1')
    n_conv1d_layer2 = kwargs.get('n_conv1d_layer2')
    size_conv1d_layer1 = kwargs.get('size_conv1d_layer1')
    size_conv1d_layer2 = kwargs.get('size_conv1d_layer2')

    # Get the input shape
    window_len = int((n_sec_before_peak + n_sec_after_peak) * sample_rate) + 1
    imu_input_shape = (window_len, 3)
    dist_input_shape = (window_len, 9)
    other_input_shape = (window_len, 1)

    if x_type == 'tuple':
        X_acc_input = Input(imu_input_shape, name='X_acc_input')
        X_gyro_input = Input(imu_input_shape, name='X_gyro_input')
        X_speed_input = Input(other_input_shape, name='X_speed_input')
        X_dist_input = Input(dist_input_shape, name='X_dist_input')
        X_distance_input = Input(other_input_shape, name='X_distance_input')
        X_ttc_input = Input(other_input_shape, name='X_ttc_input')
        X_sensor_data_input = Concatenate()([X_acc_input, X_gyro_input, X_speed_input, X_dist_input, X_distance_input,
                                             X_ttc_input])
    elif x_type == 'matrix':
        input_shape = (window_len, 3+3+1+9+1+1)  # TODO: hardcoded the dim for now, to make this auto-determined
        X_sensor_data_input = Input(input_shape, name='X_sensor_data_input')
    else:
        raise ValueError(f'The specified input type "{x_type}" is not supported.')

    # combined model after merge
    conv1d = Conv1D(filters=n_conv1d_layer1, kernel_size=size_conv1d_layer1,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(X_sensor_data_input)
    conv1d = Conv1D(filters=n_conv1d_layer2, kernel_size=size_conv1d_layer2,
                    activation='relu', padding=padding, dilation_rate=dilation_rate)(conv1d)
    flatten = Flatten()(conv1d)

    # dense layers
    dense = Dense(units=16, activation='relu', kernel_initializer='VarianceScaling')(flatten)

    # output layer
    Y = Dense(n_class, activation='softmax', name='coachable_output')(dense)

    loss = {'coachable_output': 'categorical_crossentropy'}
    if x_type == 'tuple':
        model = Model(inputs=[X_acc_input, X_gyro_input, X_speed_input, X_dist_input, X_distance_input, X_ttc_input],
                      outputs=[Y])
    elif x_type == 'matrix':
        model = Model(inputs=[X_sensor_data_input], outputs=[Y])

    return model, loss, 'adam'


if __name__ == '__main__':
    model_params = {'channels': ['ori_acc_data', 'ori_gyro_data', 'gps_data', 'dist_multilabel_data', 'tailgating_data',
                                 'fcw_data'],
                    'n_sec_before_peak': 5.0,
                    'n_sec_after_peak': 3.0,
                    'sample_rate': 10,
                    'output_X_type': 'tuple',
                    'n_class': 3,
                    'padding': 'same',
                    'dilation_rate': 1,
                    'n_conv1d_layer1': 128,
                    'n_conv1d_layer2': 64,
                    'n_conv1d_layer3': 128,
                    'n_conv1d_layer4': 64,
                    'n_conv1d_layer5': 32,
                    'size_conv1d_layer1': 5,
                    'size_conv1d_layer2': 11,
                    'size_conv1d_layer3': 5,
                    'size_conv1d_layer4': 11,
                    'size_conv1d_layer5': 21,
                    'pool_size1': 4,
                    }
    # model, _, _ = get_model_conv1d_simple(**model_params)
    # model, _, _ = get_model_conv1d_v2(**model_params)
    # model, _, _ = get_model_conv1d_v3(**model_params)
    # model, _, _ = get_model_conv1d_v3_early_fusion(**model_params)
    model, _, _ = get_model_conv1d_v3_late_fusion(**model_params)

    # model, _, _ = get_model_resnet(2000, 3, 1, 'relu', 2, 0.25, 6)
    print(model.summary())
    from tensorflow.keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='model_conv1d_v3_late_fusion.png')
