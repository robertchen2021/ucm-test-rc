from nauto_zoo import Model, ModelInput, ModelResponse
from nauto_zoo.models.utils import infer_confidence
from .per_frame import UpstreamDrowsinessDetector as UpstreamModel
from .frame_sequence import DownstreamDrowsinessDetector as DownstreamModel
import numpy as np
import caffe
from typing import Dict, Any, Optional
import os


class DrowsinessDetector(Model):
    DEFAULT_S3_MODEL_VERSION_DIR = "0.2"
    MODEL_FILES_FOLDER = "/tmp/drowsiness/"
    UPSTREAM_PROTO_FILE = MODEL_FILES_FOLDER + "UPSTREAM_PROTO"
    UPSTREAM_WEIGHTS_FILE = MODEL_FILES_FOLDER + "UPSTREAM_WEIGHTS"
    UPSTREAM_CFG_FILE = MODEL_FILES_FOLDER + "UPSTREAM_CFG"
    DOWNSTREAM_PROTO_FILE = MODEL_FILES_FOLDER + "DOWNSTREAM_PROTO"
    DOWNSTREAM_WEIGHTS_FILE = MODEL_FILES_FOLDER + "DOWNSTREAM_WEIGHTS"
    DOWNSTREAM_MEAN_FILE = MODEL_FILES_FOLDER + "DOWNSTREAM_MEAN"
    DOWNSTREAM_SCALE_FILE = MODEL_FILES_FOLDER + "DOWNSTREAM_SCALE"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        assert "threshold" in self._config
        assert "min_sequence_length" in self._config
        self.bootstrapped = False
        self._try_load()

    def bootstrap(self):
        if not self.bootstrapped:
            model_dir =  str(self._config.get("model_version", self.DEFAULT_S3_MODEL_VERSION_DIR))
            _ = self._download_from_s3
            os.makedirs(self.MODEL_FILES_FOLDER, exist_ok=True)
            _("nauto-cloud-models-test-us", "distraction_detector/"+model_dir+"/model.prototxt", self.UPSTREAM_PROTO_FILE)
            _("nauto-cloud-models-test-us", "distraction_detector/"+model_dir+"/model.caffemodel", self.UPSTREAM_WEIGHTS_FILE)
            _("nauto-cloud-models-test-us", "distraction_detector/"+model_dir+"/model.json", self.UPSTREAM_CFG_FILE)
            _("nauto-cloud-models-test-us", "distraction_detector/"+model_dir+"/downstream.prototxt", self.DOWNSTREAM_PROTO_FILE)
            _("nauto-cloud-models-test-us", "distraction_detector/"+model_dir+"/downstream.caffemodel", self.DOWNSTREAM_WEIGHTS_FILE)
            _("nauto-cloud-models-test-us", "distraction_detector/"+model_dir+"/mean.binaryproto", self.DOWNSTREAM_MEAN_FILE)
            _("nauto-cloud-models-test-us", "distraction_detector/"+model_dir+"/scale.binaryproto", self.DOWNSTREAM_SCALE_FILE)
            self._try_load()

    def _try_load(self):
        if os.path.isfile(self.UPSTREAM_PROTO_FILE):
            self.upstream_model = UpstreamModel(
                self.UPSTREAM_PROTO_FILE,
                self.UPSTREAM_WEIGHTS_FILE,
                self.UPSTREAM_CFG_FILE
            )
            self.downstream_model = DownstreamModel(
                self.DOWNSTREAM_PROTO_FILE,
                self.DOWNSTREAM_WEIGHTS_FILE,
                get_array_from_binary(self.DOWNSTREAM_MEAN_FILE),
                get_array_from_binary(self.DOWNSTREAM_SCALE_FILE)
            )
            self.bootstrapped = True

    def run(self, model_input: ModelInput, output_intermediate_results: Optional[bool] = False) -> ModelResponse:
        assert self.bootstrapped
        intermediate_results = []
        for frame in model_input.video_in:
            # batching images does not provide speedup but increases memory footprint, so processing frames one-by-one
            result = self.upstream_model.run([frame])
            intermediate_results.append(result.copy())
        intermediate_results = np.concatenate(intermediate_results, axis=-1)

        if output_intermediate_results:
            intermediate_results_ = intermediate_results.copy()

        num_frames = intermediate_results.shape[-1]
        if num_frames < self._config["min_sequence_length"]:
            num_fake_frames = self._config["min_sequence_length"] - num_frames
            num_fake_frames0 = int(num_fake_frames / 2)
            num_fake_frames1 = num_fake_frames - num_fake_frames0
            in_list = [intermediate_results,]
            if num_fake_frames0 > 0:
                if self.downstream_model.mean is not None:
                    fake_frames0 = np.repeat(self.downstream_model.mean, num_fake_frames0, axis=-1)
                else:
                    fake_frames0 = np.zeros(shape=tuple(list(intermediate_results.shape[:-1]) + [num_fake_frames0,]))
                in_list = [fake_frames0,] + in_list
            if num_fake_frames1 > 0:
                if self.downstream_model.mean is not None:
                    fake_frames1 = np.repeat(self.downstream_model.mean, num_fake_frames1, axis=-1)
                else:
                    fake_frames1 = np.zeros(shape=tuple(list(intermediate_results.shape[:-1]) + [num_fake_frames1,]))
                in_list = in_list + [fake_frames1,]
            intermediate_results = np.concatenate(in_list, axis=-1)
        elif "max_sequence_length" in self._config and num_frames > self._config["max_sequence_length"]:
            num_skip_frames = num_frames - self._config["max_sequence_length"]
            num_skip_frames0 = int(num_skip_frames / 2)
            intermediate_results = intermediate_results[num_skip_frames0 : num_skip_frames0 + self._config["max_sequence_length"]]

        result = self.downstream_model.run(intermediate_results)
        result = np.reshape(result, (2,))
        # first value is score for non-drowsiness, second is the score for drowsiness, expected to sum to 1
        score = float(result[1])

        response =  ModelResponse(
            summary="TRUE" if score > self._config["threshold"] else "FALSE",
            score=score,
            confidence=infer_confidence(score, self._config["threshold"]),
            raw_output={
                "score": score,
                "threshold": self._config["threshold"]
            }
        )

        if output_intermediate_results: 
            response.raw_output["intermediate_results"] = intermediate_results_

        return response


def get_blob_from_binary(binaryproto_filepath: str) -> np.array:
    blob_param = caffe.proto.caffe_pb2.BlobProto()
    blob_str = open(binaryproto_filepath, 'rb').read()
    blob_param.ParseFromString(blob_str)
    return blob_param


def get_array_from_binary(binaryproto_filepath: str) -> np.array:
    blobproto = get_blob_from_binary(binaryproto_filepath)
    return caffe.io.blobproto_to_array(blobproto).astype(np.float)