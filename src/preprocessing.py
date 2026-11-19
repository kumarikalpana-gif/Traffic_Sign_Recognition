
import numpy as np
from PIL import Image

from src.config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS
)


def preprocess_image(image, image_size=(IMG_WIDTH, IMG_HEIGHT)):
    """
    Convert an image to RGB, resize it, and normalize
    pixel values to the range [0, 1].

    Accepts a PIL image or an image file path.
    Returns a float32 NumPy array of shape (64, 64, 3).
    """

    if isinstance(image, (str, bytes)):
        with Image.open(image) as img:
            image = img.convert("RGB")
            image = image.resize(image_size)
            image_array = np.asarray(
                image,
                dtype=np.float32
            )
    else:
        image = image.convert("RGB")
        image = image.resize(image_size)
        image_array = np.asarray(
            image,
            dtype=np.float32
        )

    image_array = image_array / 255.0

    expected_shape = (
        IMG_HEIGHT,
        IMG_WIDTH,
        IMG_CHANNELS
    )

    if image_array.shape != expected_shape:
        raise ValueError(
            f"Unexpected image shape: {image_array.shape}"
        )

    return image_array.astype(np.float32)


def prepare_image_for_prediction(image):
    """
    Prepare one image for Keras model prediction.

    Adds a batch dimension:
    (64, 64, 3) -> (1, 64, 64, 3)
    """

    image_array = preprocess_image(image)

    return np.expand_dims(image_array, axis=0)


def preprocess_batch(images):
    """
    Preprocess a list of images and stack them into
    a batch suitable for model input.
    """

    processed_images = [
        preprocess_image(image)
        for image in images
    ]

    return np.stack(processed_images, axis=0)
