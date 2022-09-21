from nauto_zoo.models.no_motion import NoMotionDetectorModel
from nauto_zoo import ModelInput
from nauto_zoo.preprocess.sensor.combined_json import SensorPreprocessorCombinedJson
from pathlib import Path
from unittest.mock import Mock


def test_no_motion():
    preprocessor = SensorPreprocessorCombinedJson()
    preprocessed = preprocessor.preprocess_sensor_files([
        Path(__file__).parent / ".." / ".." / ".." / "test_data" / "sensor1",
        Path(__file__).parent / ".." / ".." / ".." / "test_data" / "sensor2",
        Path(__file__).parent / ".." / ".." / ".." / "test_data" / "sensor3",
        Path(__file__).parent / ".." / ".." / ".." / "test_data" / "sensor4"
    ])
    model_input = ModelInput()
    model_input.set("sensor", preprocessed)
    sut = NoMotionDetectorModel()
    sut.set_logger(Mock())
    model_response = sut.run(model_input=model_input)
    assert model_response.summary == "FALSE"
    assert model_response.raw_output == "{ 'is_no_motion': False,\n\
  'max_mov_std': { 'acc': array([1.29310852, 0.89139184, 1.74754282]),\n\
                   'gyro': array([0.03455338, 0.01947823, 0.027952  ])},\n\
  'std': { 'acc': array([1.30168739, 0.48353445, 1.10490509]),\n\
           'gyro': array([0.01440867, 0.01002347, 0.0158957 ])},\n\
  'thresholds': {'acc': 0.1, 'gyro': 0.005},\n\
  'window_size': 200}"
