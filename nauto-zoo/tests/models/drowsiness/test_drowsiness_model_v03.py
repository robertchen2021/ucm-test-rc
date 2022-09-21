import os
import json
from unittest.mock import Mock
from pathlib import Path
from nauto_zoo import ModelInput
from nauto_zoo.preprocess import VideoPreprocessorFps
import pytest


@pytest.mark.caffe
def test_drowsiness_model_v03():
    from nauto_zoo.models.drowsiness import DrowsinessDetector
    # assign event message (any message meta is fine, including none)
    with open(str(Path("./test_data/hard_brake.json").resolve()), 'r') as fh:
        event_message = json.loads(fh.read())

    # read test sensor files into a com_rec
    video_in_pb_gzip = [
        str(Path("./test_data/video-in"+str(i)+".ts").resolve()) for i in [1,2,3]
    ]
    preprocessor = VideoPreprocessorFps(fps=15)
    com_rec = preprocessor.preprocess_video_files(video_in_pb_gzip)

    # given
    model_input = ModelInput(metadata=event_message)
    model_input.set('video_in', com_rec)
    config_provided_by_ucm = {'min_sequence_length': 450,
                              'threshold': 0.8,
                              'model_version': "0.2"}
    model = DrowsinessDetector(config_provided_by_ucm)
    model.set_logger(Mock())
    model.bootstrap()

    # when
    response = model.run(model_input, output_intermediate_results=True)

    # then
    #assert response.summary == 'FALSE'
    #assert response.score == 0.
    #assert response.confidence == 100
    #assert response.summary is not None

    print("Score=", response.score)

    # "jsonify" numpy array
    response.raw_output["intermediate_results"] = response.raw_output["intermediate_results"].tolist()    
    with open(os.path.join(str(Path("./test_data").resolve()), "test_drowsiness_model_v3-out.json"), 'w') as f:
        f.write(json.dumps(response.raw_output, indent=4))
