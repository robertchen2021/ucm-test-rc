# from nauto_datasets.core.sensors import ImuStream, CombinedRecording
from pathlib import Path
from nauto_zoo.preprocess.sensor import SensorPreprocessorImu, SensorPreprocessorCombined, SensorMetadataPreprocessor
from typing import List
import numpy as np

def test_sensor_metadata():
    smp = SensorMetadataPreprocessor()
    sample_input = smp.preprocess_sensor_files(get_test_sensor_path())
    score_sum = round(np.sum(sample_input), 3)
    assert abs(score_sum - 68.751) < 0.01

def get_test_sensor_path() -> List[str]:
    return [str(Path(
        "./test_data/76d54e946eea68fa-16dc2cba08f7df35/sensor/6ad98bf91bb7930d141603a27f996e8e2481312e").resolve()),
            str(Path(
                "./test_data/76d54e946eea68fa-16dc2cba08f7df35/sensor/27f5c0686f10c4f4811b813544309571d4f911c6").resolve()),
            str(Path(
                "./test_data/76d54e946eea68fa-16dc2cba08f7df35/sensor/86f7ed4b159568a8d68fa5d6a8e7f957eb76e53c").resolve()),
            str(Path(
                "./test_data/76d54e946eea68fa-16dc2cba08f7df35/sensor/b5566e51fac81f46caf7c2147a26f841d5ec5521").resolve()),
            str(Path(
                "./test_data/76d54e946eea68fa-16dc2cba08f7df35/sensor/d136707085011b9017c8130a3ec73b872c69237d").resolve())]
