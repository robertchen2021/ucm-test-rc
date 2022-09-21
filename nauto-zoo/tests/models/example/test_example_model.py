from unittest.mock import Mock
import numpy as np

from nauto_zoo import ModelInput
from nauto_zoo.models.example.example_model import ExampleModel


def test_example_model():
    # given
    model_input = ModelInput()
    model_input.set('snapshot_in', [np.random.random(100)])
    config_provided_by_ucm = {'confidence_threshold': 90}
    model = ExampleModel(config_provided_by_ucm)
    model.set_logger(Mock())

    # when
    response = model.run(model_input)

    # then
    assert response.summary == 'TRUE'
    assert response.score == 1.
    assert response.confidence == 100
    assert response.summary is not None
