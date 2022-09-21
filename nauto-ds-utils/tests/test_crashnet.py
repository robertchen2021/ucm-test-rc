from nauto_ds_utils.preprocessing.crashnet import CrashnetSensorPreprocessor
from nauto_ds_utils.utils.tf import load_model
from nauto_ds_utils.utils.sensor import get_combined_recording

from numpy import squeeze, array
from tensorflow.keras.utils import to_categorical
import os
import unittest


class TestCrashnet(unittest.TestCase):
    """
    This should be ran from the parent directory:
    >> cd nauto-ai/nauto-ds-utils/
    >> pytest .
    """
    @classmethod
    def setUpClass(cls):
        data_directory = 'tests/data'
        cls.model_path = 'crashnet_v15.hdf5'
        cls.sensor_files = ['16a348c2633af146/b795f92a7ccff4f8615ac5d2c8e445d105abe777',
                            '16a348c2633af146/de303743025b6842407768a024363785d6247fd2',
                            '16a348c2633af146/0040f5c4cce02ee7c3a70937d2878eacb7360d09']
        cls.model_path = os.path.join(data_directory, cls.model_path)
        cls.sensor_files = [os.path.join(data_directory, x) for x in cls.sensor_files]

        cls.window_size = 4000
        cls.num_vehicle_types = 15
        cls.use_oriented = False
        cls.vehicle_type_enum = 0

    def test_load_model(self):
        crashnet = load_model(self.model_path)
        self.assertTrue(crashnet is not None)

    def test_preprocessing(self):
        sensor_data = (CrashnetSensorPreprocessor(window_size=self.window_size,
                                                  use_oriented=self.use_oriented)
                       .preprocess_sensor_files(self.sensor_files)
                       )

        self.assertTrue(sensor_data.shape == (1, self.window_size, 6, 1))

    def test_prediction(self):
        model = load_model(self.model_path)
        sensor_data = (CrashnetSensorPreprocessor(window_size=self.window_size,
                                                  use_oriented=self.use_oriented)
                       .preprocess_sensor_files(self.sensor_files)
                       )
        vt = [to_categorical(array([self.vehicle_type_enum]),
                             num_classes=self.num_vehicle_types)]

        result = [squeeze(model.predict([_input1, _input2], batch_size=1)).tolist()[1]
                  for _input1, _input2 in zip([sensor_data], vt)]
        self.assertTrue(array(result).shape == (1, 2))


if __name__ == '__main__':
    unittest.main()
