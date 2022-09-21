from .interfaces import AbstractSensorPreprocessor
from .combined import SensorPreprocessorCombined
from typing import List, Dict, Any
from nauto_datasets.serialization.jsons.sensors import combined_recording_to_json


class SensorPreprocessorCombinedJson(AbstractSensorPreprocessor):
    def __init__(self):
        self._preprocessor_combined = SensorPreprocessorCombined()

    def preprocess_sensor_files(self, sensor_files: List[str], metadata: Dict = None) -> Dict[str, Any]:
        combined = self._preprocessor_combined.preprocess_sensor_files(sensor_files)
        return combined_recording_to_json(combined)
