from tcn import TCN
from tensorflow.keras.layers import Concatenate, Dropout, Dense, Flatten, Input, MaxPooling1D
from tensorflow.keras.models import Model


def get_model_tcn_v1(**kwargs):
    """
    Start with 2 TCN blocks for each head, merge, then add another 2 TCN blocks
    """
    # Get params
    n_class = kwargs.get('n_class')
    x_type = kwargs.get('output_X_type', 'tuple')
    n_sec_before_peak = kwargs.get('n_sec_before_peak')
    n_sec_after_peak = kwargs.get('n_sec_after_peak')
    sample_rate = kwargs.get('sample_rate')
    tcn_nb_filters = kwargs.get('tcn_nb_filters')
    tcn_nb_stacks = kwargs.get('tcn_nb_stacks')
    tcn_kernel_size = kwargs.get('tcn_kernel_size')
    tcn_dilations = kwargs.get('tcn_dilations')
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
    tcn = TCN(nb_filters=tcn_nb_filters, kernel_size=tcn_kernel_size, nb_stacks=tcn_nb_stacks, dilations=tcn_dilations,
              padding='causal', return_sequences=True)(X_acc_input)
    dropout_acc = Dropout(0.5)(tcn)

    # sub-model for X_gyro
    tcn = TCN(nb_filters=tcn_nb_filters, kernel_size=tcn_kernel_size, nb_stacks=tcn_nb_stacks, dilations=tcn_dilations,
              padding='causal', return_sequences=True)(X_gyro_input)
    dropout_gyro = Dropout(0.5)(tcn)

    # sub-model for X_speed
    tcn = TCN(nb_filters=tcn_nb_filters, kernel_size=tcn_kernel_size, nb_stacks=tcn_nb_stacks, dilations=tcn_dilations,
              padding='causal', return_sequences=True)(X_speed_input)
    dropout_speed = Dropout(0.5)(tcn)

    # sub-model for X_dist
    tcn = TCN(nb_filters=tcn_nb_filters, kernel_size=tcn_kernel_size, nb_stacks=tcn_nb_stacks, dilations=tcn_dilations,
              padding='causal', return_sequences=True)(X_dist_input)
    dropout_dist = Dropout(0.5)(tcn)

    # sub-model for X_distance
    tcn = TCN(nb_filters=tcn_nb_filters, kernel_size=tcn_kernel_size, nb_stacks=tcn_nb_stacks, dilations=tcn_dilations,
              padding='causal', return_sequences=True)(X_distance_input)
    dropout_distance = Dropout(0.5)(tcn)

    # sub-model for X_ttc
    tcn = TCN(nb_filters=tcn_nb_filters, kernel_size=tcn_kernel_size, nb_stacks=tcn_nb_stacks, dilations=tcn_dilations,
              padding='causal', return_sequences=True)(X_ttc_input)
    dropout_ttc = Dropout(0.5)(tcn)

    # merge
    merged = Concatenate()([dropout_acc, dropout_gyro, dropout_speed, dropout_dist, dropout_distance, dropout_ttc])

    # combined model after merge
    tcn = TCN(nb_filters=tcn_nb_filters, kernel_size=tcn_kernel_size, nb_stacks=tcn_nb_stacks, dilations=tcn_dilations,
              padding='causal', return_sequences=True)(merged)
    dropout = Dropout(0.5)(tcn)
    maxpool1d = MaxPooling1D(pool_size=pool_size1)(dropout)
    flatten = Flatten()(maxpool1d)

    # dense layers
    merged_output = Dropout(0.4)(flatten)
    merged_output = Dense(units=64, activation='relu')(merged_output)

    # output layer
    Y = Dense(n_class, activation='softmax', name='coachable_output')(merged_output)

    loss = {'coachable_output': 'categorical_crossentropy'}
    model = Model(inputs=[X_acc_input, X_gyro_input, X_speed_input, X_dist_input, X_distance_input, X_ttc_input],
                  outputs=[Y])

    return model, loss, 'adam'


def get_model_tcn_simple(**kwargs):
    # Get params
    n_class = kwargs.get('n_class')
    x_type = kwargs.get('output_X_type', 'tuple')
    n_sec_before_peak = kwargs.get('n_sec_before_peak')
    n_sec_after_peak = kwargs.get('n_sec_after_peak')
    sample_rate = kwargs.get('sample_rate')
    tcn_nb_stacks = kwargs.get('tcn_nb_stacks')
    tcn_kernel_size = kwargs.get('tcn_kernel_size')
    tcn_dilations = kwargs.get('tcn_dilations')

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

    tcn_layer = TCN(kernel_size=tcn_kernel_size, nb_stacks=tcn_nb_stacks, dilations=tcn_dilations, padding='causal')\
        (X_sensor_data_input)
    dense = Dense(units=16, activation='relu', kernel_initializer='VarianceScaling')(tcn_layer)
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
                    'tcn_nb_filters': 64,
                    'tcn_nb_stacks': 2,
                    'tcn_kernel_size': 2,
                    'tcn_dilations': (1,2,4,8),
                    'pool_size1': 4,
                    }
    # model, _, _ = get_model_tcn_simple(**model_params)
    model, _, _ = get_model_tcn_v1(**model_params)
    print(model.summary())
    from tensorflow.keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='model_tcn_v1.png')
