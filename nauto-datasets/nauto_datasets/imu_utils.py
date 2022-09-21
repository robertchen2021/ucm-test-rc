import numpy as np
from typing import List, NamedTuple, Tuple, Optional
from nauto_datasets.core.sensors import ImuStream, ComXYZStream, XYZStream, CombinedRecording, DeviceOrientationStream

GRAVITY = -9.81


def build_R_i2v(dev_or_stream: DeviceOrientationStream) -> np.ndarray:
    """
    Builds orthogonal rotation matrix (IMU to Vehicle frame) from device orientation stream
    device orientation stream contains converged device orientation under VotingResult key
    each sensor recording provides atleast one votingResult solution, combined recording thus contains
    multiple solutions. This function utilizes the latest solution to build the rotation matrix
    """
    orientation_data = np.empty((0, 4))
    voting_sensor_ns = []
    theta_x = []
    theta_y = []
    theta_z = []

    if 'votingResult' not in dev_or_stream.name.tolist():
        raise ValueError("R_i2v cannot be computed, orientation has not converged!")

    for idx, name in enumerate(dev_or_stream.name.tolist()):
        if name == 'votingResult':
            orientation_data = np.append(orientation_data, np.array([[dev_or_stream.sensor_ns[idx],
                                                                      dev_or_stream.theta_x[idx],
                                                                      dev_or_stream.theta_y[idx],
                                                                      dev_or_stream.theta_z[idx]]]), axis=0)
            voting_sensor_ns.append(dev_or_stream.sensor_ns[idx])
            theta_x.append(dev_or_stream.theta_x[idx])
            theta_y.append(dev_or_stream.theta_y[idx])
            theta_z.append(dev_or_stream.theta_z[idx])

    # Get the latest solution
    latest_idx = np.argmax(orientation_data[:, 0])
    orientation_data = orientation_data[latest_idx:latest_idx + 1, :]

    if orientation_data.shape[0] != 1:
        raise ValueError("R_i2v cannot be computed, multiple solution available!")
    else:
        theta_x = orientation_data[0, 1]
        theta_y = orientation_data[0, 2]
        theta_z = orientation_data[0, 3]

    # Compute rotation matrices for X, Y, and Z axes
    matX = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    matY = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    matZ = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    # Build rotation matrix for sequence X->Y->Z
    R_i2v = np.matmul(matZ, np.matmul(matY, matX))

    return R_i2v


def rotate_acc_gyro_array(acc_array: np.ndarray, gyro_array: np.ndarray, R: np.ndarray) -> Optional[Tuple[np.ndarray]]:
    """
    Rotates ACC and GYRO streams with rotation matrix R
    R can be IMU-to-Vehicle or Vehicle-to-IMU
    """
    return np.matmul(R, acc_array), np.matmul(R, gyro_array)


def convert_oriented_to_raw(com_rec: CombinedRecording,
                            acc_field='rt_oriented_acc',
                            gyro_field='rt_oriented_gyro',
                            orientation_field='rt_device_orientation',
                            components=['sensor_ns', 'system_ms', 'x', 'y', 'z']
                            ) -> Optional[Tuple[ComXYZStream]]:
    """
    Converts oriented_streams to raw using device orientation stream
    For converting oriented to raw, gravity has be subtracted before rotating streams back to device frame.
    """
    if getattr(com_rec, acc_field).stream._is_empty() or getattr(com_rec, gyro_field).stream._is_empty():
        raise ValueError("Cannot get raw from oriented, oriented streams are empty!")
    else:
        # Compute IMU to Vehicle rotation matrix from device orientation stream
        R_i2v = build_R_i2v(dev_or_stream=getattr(com_rec, orientation_field).stream)

        # Extract IMU streams
        rt_oriented_acc = np.c_[[getattr(getattr(com_rec, acc_field).stream, field) for field in components]]
        rt_oriented_gyro = np.c_[[getattr(getattr(com_rec, gyro_field).stream, field) for field in components]]

        # Subtract gravity since oriented streams are gravity compensated
        rt_oriented_acc[-1, :] = rt_oriented_acc[-1, :] - GRAVITY
        # Rotate streams to device frame
        rt_acc_converted, rt_gyro_converted = rotate_acc_gyro_array(acc_array=rt_oriented_acc[2:, ],
                                                                    gyro_array=rt_oriented_gyro[2:, ], R=R_i2v.T)

        rt_acc_converted_stream = XYZStream(
            sensor_ns=rt_oriented_acc[0, :],
            system_ms=rt_oriented_acc[1, :],
            x=rt_acc_converted[0, :],
            y=rt_acc_converted[1, :],
            z=rt_acc_converted[2, :],
            w=np.array([]))

        rt_gyro_converted_stream = XYZStream(
            sensor_ns=rt_oriented_gyro[0, :],
            system_ms=rt_oriented_gyro[1, :],
            x=rt_gyro_converted[0, :],
            y=rt_gyro_converted[1, :],
            z=rt_gyro_converted[2, :],
            w=np.array([]))

    return ComXYZStream.from_substreams([rt_acc_converted_stream]), ComXYZStream.from_substreams([rt_gyro_converted_stream])


def convert_raw_to_oriented(com_rec: CombinedRecording,
                            acc_field='rt_acc',
                            gyro_field='rt_gyro',
                            orientation_field='rt_device_orientation',
                            components=['sensor_ns', 'system_ms', 'x', 'y', 'z']
                            ) -> Optional[Tuple[ComXYZStream]]:
    """
    Converts oriented_streams to raw using device orientation stream
    While converting raw to oriented, gravity has to be added after
    rotating the streams to vehicle frame to compute gravity compensated oriented streams.
    """
    if getattr(com_rec, acc_field).stream._is_empty() or getattr(com_rec, gyro_field).stream._is_empty():
        raise ValueError("Cannot get oriented from raw, raw streams are empty!")
    else:
        # Compute IMU to Vehicle rotation matrix from device orientation stream
        R_i2v = build_R_i2v(dev_or_stream=getattr(com_rec, orientation_field).stream)

        # Extract IMU streams
        rt_acc = np.c_[[getattr(getattr(com_rec, acc_field).stream, field) for field in components]]
        rt_gyro = np.c_[[getattr(getattr(com_rec, gyro_field).stream, field) for field in components]]

        # Rotate streams to vehicle frame
        rt_oriented_acc_converted, rt_oriented_gyro_converted = rotate_acc_gyro_array(acc_array=rt_acc[2:, ],
                                                                                              gyro_array=rt_gyro[2:, ],
                                                                                              R=R_i2v)
        # Add gravity to make the oriented streams gravity compensated
        rt_oriented_acc_converted[-1, :] = rt_oriented_acc_converted[-1, :] + GRAVITY

        rt_oriented_acc_converted_stream = XYZStream(
            sensor_ns=rt_acc[0, :],
            system_ms=rt_acc[1, :],
            x=rt_oriented_acc_converted[0, :],
            y=rt_oriented_acc_converted[1, :],
            z=rt_oriented_acc_converted[2, :],
            w=np.array([]))

        rt_oriented_gyro_converted_stream = XYZStream(
            sensor_ns=rt_gyro[0, :],
            system_ms=rt_gyro[1, :],
            x=rt_oriented_gyro_converted[0, :],
            y=rt_oriented_gyro_converted[1, :],
            z=rt_oriented_gyro_converted[2, :],
            w=np.array([]))

    return ComXYZStream.from_substreams([rt_oriented_acc_converted_stream]), ComXYZStream.from_substreams([rt_oriented_gyro_converted_stream])
