from typing import List
import numpy as np
import caffe
import cv2
import json


class UpstreamDrowsinessDetector(object):
    """
    This model produces distraction score for individual frame (image)
    """
    def __init__(self, model_path: str, weights_path: str, cfg_path: str):
        cfg = None
        with open(cfg_path) as f:
            cfg = json.loads(f.read())
        assert(cfg is not None)
        input_scale = cfg["input_scale"] if "input_scale" else None
        if input_scale is not None:
            assert(isinstance(input_scale, (float, int, list, tuple)))
            if isinstance(input_scale, list):
                assert(len(input_scale) in [1,3]) #Scalar or 3-channel
                if len(input_scale) == 1: input_scale = input_scale[0]
                else: input_scale = np.array(input_scale)
        mean = cfg["mean"] if "mean" else None
        if mean is not None:
            assert(isinstance(mean, (float, int, list, tuple)))
            if isinstance(mean, list):
                assert(len(mean) in [1,3]) #Scalar or 3-channel
                mean = np.array(mean)
        channel_swap = cfg["channel_swap"] if "channel_swap" else None
        if channel_swap is not None:
            assert(isinstance(channel_swap, (list, tuple)))
            assert(len(channel_swap) == 3) #3-channel
        self.net = caffe.Classifier(model_path, weights_path, mean=mean,
                         channel_swap=channel_swap, input_scale=input_scale)
        self.image_resize = cfg["image_resize"] if "image_resize" else None
        if self.image_resize is not None:
            assert(isinstance(self.image_resize, (list, tuple)))
            assert(len(self.image_resize) == 2) #(W,H)
        self.image_crop = cfg["image_crop"] if "image_crop" else None
        if self.image_crop is not None:
            assert(isinstance(self.image_crop, (list, tuple)))
            assert(len(self.image_crop) == 4) #(X,Y,W,H)
        self.output_node = cfg["output_node"] if "output_node" else "pool6"

    def run(self, image_arrays: np.array) -> np.array:
        #Input is RGB (H,W,C) [0,255], where WxH is native video resolution
        self.FF([self.preprocess(image_array) for image_array in image_arrays])
        return self.net.blobs[self.output_node].data

    def preprocess(self, image_array: np.array) -> np.array:
        if self.image_resize is not None:
            #image_array = cv2.resize(image_array, (self.image_resize[0], self.image_resize[1]))            
            image_array = cv2.resize(image_array.astype(np.float32), (self.image_resize[0], self.image_resize[1])) #For consistency with Caffe io.py
        if self.image_crop is not None:
            image_array = image_array[self.image_crop[1]:self.image_crop[1]+self.image_crop[3],self.image_crop[0]:self.image_crop[0]+self.image_crop[2],:]
        return image_array.astype(np.float32)

    def FF(self, inputs: List[np.array]) -> np.array:
        input_ = np.zeros(
            (len(inputs), self.net.image_dims[0], self.net.image_dims[1], inputs[0].shape[2]),
            dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = cv2.resize(in_, (self.net.image_dims[1], self.net.image_dims[0]))

        center = np.array(self.net.image_dims) / 2.0
        crop = (np.tile(center, (1, 2))[0] + np.concatenate([
            -self.net.crop_dims / 2.0,
            self.net.crop_dims / 2.0
        ])).astype(np.int)
        input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]
        caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]], dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.net.transformer.preprocess(self.net.inputs[0], in_)
        
        # Reshape input blob to match the current input
        self.net.blobs[self.net.inputs[0]].reshape(*caffe_in.shape)
        return self.net.forward_all(**{self.net.inputs[0]: caffe_in})