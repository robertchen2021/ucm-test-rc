import numpy
from nauto_zoo.preprocess.image import ImagePreprocessor, INTERPOLATION_AREA, CHANNEL_ORDER_BGR, CHANNEL_ORDER_RGB
from pathlib import Path


def path_to_test_data(file: str) -> str:
    return str(Path(f"./test_data/{file}").resolve())


def test_reads_images_empty_config():
    sut = ImagePreprocessor()
    images = sut.preprocess_image_batch([
        path_to_test_data('snapshot-in1.jpg'),
        path_to_test_data('snapshot-in2.jpg'),
        path_to_test_data('snapshot-in3.jpg')
    ])
    assert len(images) == 3
    for image in images:
        assert image is not None
        assert image.shape == (360, 640, 3)


def test_reads_images_and_resizes():
    sut = ImagePreprocessor(width=100, height=200)
    images = sut.preprocess_image_batch([
        path_to_test_data('snapshot-in1.jpg'),
        path_to_test_data('snapshot-in2.jpg'),
        path_to_test_data('snapshot-in3.jpg')
    ])
    assert len(images) == 3
    for image in images:
        assert image is not None
        assert image.shape == (200, 100, 3)


def test_reads_images_and_resizes_with_interpolation():
    sut = ImagePreprocessor(width=100, height=200, interpolation=INTERPOLATION_AREA)
    images = sut.preprocess_image_batch([
        path_to_test_data('snapshot-in1.jpg'),
        path_to_test_data('snapshot-in2.jpg'),
        path_to_test_data('snapshot-in3.jpg')
    ])
    assert len(images) == 3
    for image in images:
        assert image is not None
        assert image.shape == (200, 100, 3)


def test_converts_to_rgb():
    sut_rgb = ImagePreprocessor(channel_order=CHANNEL_ORDER_RGB)
    sut_bgr = ImagePreprocessor(channel_order=CHANNEL_ORDER_BGR)
    rgb_image = sut_rgb.preprocess_image(path_to_test_data('snapshot-in1.jpg'))
    bgr_image = sut_bgr.preprocess_image(path_to_test_data('snapshot-in1.jpg'))
    rgb_image = rgb_image.reshape((rgb_image.shape[0] * rgb_image.shape[1], rgb_image.shape[2]))
    bgr_image = bgr_image.reshape((bgr_image.shape[0] * bgr_image.shape[1], bgr_image.shape[2]))
    for i in range(rgb_image.shape[0]):
        rgb = rgb_image[i]
        bgr = bgr_image[i]
        assert rgb[0] == bgr[2]
        assert rgb[1] == bgr[1]
        assert rgb[2] == bgr[0]


def test_conversion_to_rgb_by_default():
    sut_default = ImagePreprocessor()
    sut_rgb = ImagePreprocessor(channel_order=CHANNEL_ORDER_RGB)
    image_default = sut_default.preprocess_image(path_to_test_data('snapshot-in1.jpg'))
    image_rgb = sut_rgb.preprocess_image(path_to_test_data('snapshot-in1.jpg'))
    assert numpy.array_equal(image_default, image_rgb)
