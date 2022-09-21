import caffe
from typing import List
import numpy as np


class DownstreamDrowsinessDetector(object):
    """
    This model produces distraction score for frame sequence (video)
    """
    def __init__(self, model_path: str, weights_path: str, mean: np.array, scale: np.array, output_node: str = "prob"):
        self.net = caffe.Net(model_path, weights_path, caffe.TEST)
        self.output_node = output_node
        self.mean = mean
        self.scale = scale

    def run(self, input_data: np.array) -> np.array:
        self.FF(self.preprocess(input_data))
        return self.net.blobs[self.output_node].data

    def preprocess(self, input_data: np.array) -> np.array:
        input_data -= self.mean
        input_data *= self.scale
        return input_data

    def FF(self, inputs: List[np.array]) -> np.array:
        for ix, in_ in enumerate(inputs):
            assert (in_.shape == inputs[0].shape)
        caffe_in = np.zeros((len(inputs),) + tuple(inputs[0].shape), dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            caffe_in[ix] = in_
        self.net.blobs[self.net.inputs[0]].reshape(*caffe_in.shape)
        return self.net.forward_all(**{self.net.inputs[0]: caffe_in})
