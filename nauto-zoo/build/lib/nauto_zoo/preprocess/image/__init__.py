import abc
import numpy as np
from typing import List, Optional
import cv2


KERAS_APPLICATION_MOBILENET = "mobilenet"
INTERPOLATION_NEAREST = "nearest"
INTERPOLATION_LINEAR = "linear"
INTERPOLATION_AREA = "area"
INTERPOLATION_CUBIC = "cubic"
INTERPOLATION_LANCZOS4 = "lanczos4"
CHANNEL_ORDER_RGB = "RGB"
CHANNEL_ORDER_BGR = "BGR"

INTERPOLATIONS = {
    INTERPOLATION_NEAREST: cv2.INTER_NEAREST,
    INTERPOLATION_LINEAR: cv2.INTER_LINEAR,
    INTERPOLATION_AREA: cv2.INTER_AREA,
    INTERPOLATION_CUBIC: cv2.INTER_CUBIC,
    INTERPOLATION_LANCZOS4: cv2.INTER_LANCZOS4
}


class AbstractImagePreprocessor(abc.ABC):
    @abc.abstractmethod
    def preprocess_image(self, image_file_name: str) -> np.ndarray:
        pass

    def preprocess_image_batch(self, image_file_names: List[str]) -> List[np.array]:
        return [self.preprocess_image(image_file_name) for image_file_name in image_file_names]


class KerasImagePreprocessor(AbstractImagePreprocessor):
    def __init__(self, keras_application: Optional[str]=None):
        if keras_application == KERAS_APPLICATION_MOBILENET:
            from tensorflow.keras.applications.mobilenet import preprocess_input
            self.preprocess_fn = preprocess_input
        elif keras_application is None:
            self.preprocess_fn = lambda image: image
        else:
            raise RuntimeError(f"Unknown keras application `{keras_application}`")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        return self.preprocess_fn(np.expand_dims(image, axis=0))


class ImagePreprocessor(AbstractImagePreprocessor):
    def __init__(
            self,
            width: Optional[int]=None,
            height: Optional[int]=None,
            interpolation: str=INTERPOLATION_CUBIC,
            channel_order: str=CHANNEL_ORDER_RGB,
            keras_application: Optional[str]=None
    ):
        if width is not None and height is not None:
            self._resize_fn = lambda image: cv2.resize(
                image,
                (width, height),
                interpolation=INTERPOLATIONS[interpolation]
            )
        else:
            self._resize_fn = lambda image: image

        if channel_order == CHANNEL_ORDER_RGB:
            self._color_convert_fn = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif channel_order == CHANNEL_ORDER_BGR:
            self._color_convert_fn = lambda image: image
        else:
            raise RuntimeError(f"Unknown channel order `{channel_order}`")

        if keras_application is not None:
            self._keras_preprocess_fn = KerasImagePreprocessor(keras_application=keras_application).preprocess_image
        else:
            self._keras_preprocess_fn = lambda image: image

    def preprocess_image(self, image_file_name: str) -> np.ndarray:
        image = cv2.imread(image_file_name)
        image = self._resize_fn(image)
        image = self._color_convert_fn(image)
        return self._keras_preprocess_fn(image)


class ImagePreprocessorSaveFiles(AbstractImagePreprocessor):
    def preprocess_image_batch(self, image_file_names: List[str]) -> List[str]:
        return image_file_names

    def preprocess_image(self, image_file_name: str) -> str:
        return image_file_name
