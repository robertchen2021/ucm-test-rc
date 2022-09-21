from typing import List, Optional, Dict

import numpy as np

from nauto_ds_utils.utils.sensor import get_combined_recording

class RawIMUPreprocess(object):
    """Preprocess Sensors (Non-Omnifusion)
    1. [IMU Stream]: Access the desired IMU field via name (acc/gyro/gps field)
        a. Also, give the desired IMU components [sensor_ns, x, y, z] and etc
    2. [Chronological]: sort by sensor_ns 
    3. [Optional] Orient Sensor Stream: (creates pitch_angle and angle)
    4. [Optional] Trim Sensor Stream: (bou)

    fields: 
    acc
    gyro
    gps
    acc, gyro, and gps field names
    """

    def __init__(self,
                 protos: List[str],
                 acc_field: str = "rt_acc",
                 gyro_field: str = "rt_gyro",
                 gps_field: str = "gps",
                 orient_sensor: bool = True,
                 start_ns: Optional[int] = None,
                 end_ns: Optional[int] = None,
                 imu_components: List[str] = ['sensor_ns', 'x', 'y', 'z'],
                 gps_components: List[str] = ['sensor_ns', 'speed'],
                 to_utc_time: bool = True,
                 window_length: Optional[int] = 5
                 ):
        # self.sensor_protos = protos
        self.acc_field = acc_field
        self.gyro_field = gyro_field
        self.gps_field = gps_field
        self.windowLength = window_length
        self.orient_sensor = orient_sensor
        self.start_ns = start_ns
        self.end_ns = end_ns
        if protos is None or len(protos) < 1:
            raise ValueError('Missing Protobufs!')

        self._preprocess_imu(protos=protos,
                             imu_components=imu_components,
                             gps_components=gps_components,
                             to_utc_time=to_utc_time,
                             orient_sensor=orient_sensor,
                             start_ns=start_ns,
                             end_ns=end_ns)

    def _sort_sensor(self,
                     com_rec,
                     field: str,
                     components: List[str]) -> np.ndarray:
        """Returns the sensor stream ordered by sensor_ns"""
        tmp = np.c_[[getattr(getattr(com_rec, field).stream, x)
                     for x in components]]
        return tmp.T[tmp[0, :].argsort()]

    def get_angle(self) -> None:
        """This computes the angle (degrees) and phi_xz (radians between x and
         z axes). This creates two new attributes: `angle` and `phi_xz`. 57.3 
         is a numerical optimization for 180 / pi."""

        g_xz = np.sqrt(np.median(np.square(self.acc[:, 1]) +
                                 np.square(self.acc[:, 3])))
        arg = np.median(self.acc[:, 3] / g_xz)

        if (arg > 1) | (arg < -1):
            phixz = np.arccos(np.sign(arg))
        else:
            phixz = np.arccos(arg)

        self.angle = phixz * 57.3
        self.phi_xz = phixz

    def get_rot_mat(self) -> np.ndarray:
        """Creates a (4, 4) rotational matrix"""
        m = np.identity(4)
        m[1][1], m[1][3] = np.cos(self.phi_xz), -np.sin(self.phi_xz)
        m[3][1], m[3][3] = -m[1][3], m[1][1]
        return m

    def _trim_sensors(self, start_ns: int, end_ns: int) -> None:
        """Trimming criteria is based on starting and ending nanoseconds"""
        self.acc = self.acc[(self.acc[:, 0] >= start_ns)
                            & (self.acc[:, 0] <= end_ns)]
        self.gyro = self.gyro[(self.gyro[:, 0] >= start_ns)
                              & (self.gyro[:, 0] <= end_ns)]
        self.gps = self.gps[(self.gps[:, 0] >= start_ns - 1e9)
                            & (self.gps[:, 0] <= end_ns + 1e9)]

    def _preprocess_imu(self,
                        protos: List[str],
                        imu_components: List[str],
                        gps_components: List[str],
                        to_utc_time: bool,
                        orient_sensor: bool,
                        start_ns: Optional[int] = None,
                        end_ns: Optional[int] = None) -> None:
        """Instantiate acc, gyro, and gps stream. This is the main sensor 
        preprocessing class method."""
        if protos is None:
            raise TypeError('Missing sensor files!')

        if to_utc_time:
            com_rec = get_combined_recording(
                paths=protos, max_pool_connection=100).to_utc_time()
        else:
            com_rec = get_combined_recording(
                paths=protos, max_pool_connection=100)

        self.acc = self._sort_sensor(com_rec, self.acc_field, imu_components)
        self.gyro = self._sort_sensor(com_rec, self.gyro_field, imu_components)
        self.gps = self._sort_sensor(com_rec, self.gps_field, gps_components)

        if orient_sensor is True:
            self.orient_sensors()

    def orient_sensors(self):
        """Orients the sensor streams. It's up to the user to keep this 
        consistent. This will also create """
        self.get_angle()
        m = self.get_rot_mat().T
        self.acc = self.acc.dot(m)
        self.gyro = self.gyro.dot(m)

    def compute_acc_features(self,
                             imu_components,
                             agg_func=np.mean,
                             sample_rate: int = 200) -> Dict[str, float]:
        """Compute Acceloremeter Features"""
        tmp = {}
        tmp['n_acc_sample'] = len(self.acc)
        for i, component in enumerate(imu_components):
            tmp[f'{component}_acc_{agg_func.__name__}'] = \
                float(agg_func(self.acc[:, i+1]))
            tmp[f'{component}_acc_total'] = \
                float(np.abs(np.sum(self.acc[:, i+1])) / sample_rate)
            tmp[f'{component}_acc_var'] = float(np.var(self.acc[:, i+1]))
        return tmp

    def compute_gyro_features(self,
                              imu_components: List[str],
                              agg_func=np.mean,
                              sample_rate: int = 200) -> Dict[str, float]:
        """Compute Gyroscope Features"""
        tmp = {}
        tmp['n_gyro_sample'] = len(self.gyro)
        for i, component in enumerate(imu_components):
            tmp[f'{component}_gyro_{agg_func.__name__}'] = \
                float(agg_func(self.gyro[:, i+1]))
            tmp[f'{component}_angle'] = \
                float(np.abs(np.sum(self.gyro[:, i+1]) / sample_rate) * 57.3)
            tmp[f'{component}_gyro_var'] = float(np.var(self.gyro[:, i+1]))
        return tmp

    def compute_gps_features(self,
                             gps_components: List[str],
                             agg_func=np.mean) -> Dict[str, float]:
        """Compute GPS Features"""
        tmp = {}
        tmp['n_gps_sample'] = len(self.gps)

        for i, component in enumerate(gps_components):
            tmp[f'{component}_{agg_func.__name__}'] = \
                float(agg_func(self.gps[:, i+1]))
        return tmp

    def to_feature_vector(self, agg_func=np.mean,
                          sample_rate: int = 200,
                          imu_components=['x', 'y', 'z'],
                          gps_components=['speed']) -> Dict[str, float]:
        """Computes the feature vector. This will be 

        To be consistent, we will use a fixed window length than a dynamic
        duration:
        x_acc, y_acc, z_acc,
        x_acc_total, y_acc_total, z_acc_total,
        x_gyro, y_gyro, z_gyro,
        x_angle, y_angle, z_angle
        """
        if self.start_ns is None:
            raise ValueError('Requires Start sensor ns!')

        if self.end_ns is None:
            self.end_ns = self.start_ns + self.windowLength * 1e9

        try:
            self._trim_sensors(start_ns=self.start_ns, end_ns=self.end_ns)

            result = {}
            result['pitch'] = self.angle if self.orient_sensor else None
            result.update(
                self.compute_acc_features(imu_components=imu_components,
                                          agg_func=agg_func,
                                          sample_rate=sample_rate))
            result.update(
                self.compute_gyro_features(imu_components=imu_components,
                                           agg_func=agg_func,
                                           sample_rate=sample_rate))
            result.update(
                self.compute_gps_features(gps_components=gps_components,
                                          agg_func=agg_func))
            return result
        except e:
            return e
