from .interfaces import AbstractSensorPreprocessor
from typing import List, Dict


class SensorPreprocessorSaveFiles(AbstractSensorPreprocessor):
    def preprocess_sensor_files(self, sensor_files: List[str], metadata: Dict = None):
        return sensor_files
