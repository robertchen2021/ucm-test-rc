from typing import List, Any, Dict
import abc


class AbstractSensorPreprocessor(abc.ABC):
    @abc.abstractmethod
    def preprocess_sensor_files(self, sensor_files: List[str], metadata: Dict = None) -> Any:
        pass
