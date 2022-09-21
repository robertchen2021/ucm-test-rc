from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM
from tensorflow.keras.models import Model
# from tensorflow.keras import initializers


def get_model_lstm_simple(**kwargs):
    # Get params
    n_class = kwargs.get('n_class')
    x_type = kwargs.get('output_X_type', 'tuple')
    n_sec_before_peak = kwargs.get('n_sec_before_peak')
    n_sec_after_peak = kwargs.get('n_sec_after_peak')
    sample_rate = kwargs.get('sample_rate')
    n_lstm_layer1 = kwargs.get('n_lstm_layer1')

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

    lstm = LSTM(n_lstm_layer1, return_sequences=False, recurrent_dropout=0.5)(X_sensor_data_input)
    dense = Dense(units=16, activation='relu', kernel_initializer='VarianceScaling')(lstm)
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
                    'n_lstm_layer1': 64,
                    }
    model, _, _ = get_model_lstm_simple(**model_params)
    print(model.summary())
    from tensorflow.keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='model_lstm_simple.png')
