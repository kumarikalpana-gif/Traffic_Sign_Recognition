
import pandas as pd
import numpy as np
from PIL import Image

from src.config import (
    TRAIN_METADATA,
    VALIDATION_METADATA,
    TEST_METADATA,
    CLASS_METADATA
)


def load_metadata(split="train"):
    """
    Load metadata for a dataset split.

    Supported splits: train, validation, test, classes.
    """

    metadata_paths = {
        "train": TRAIN_METADATA,
        "validation": VALIDATION_METADATA,
        "test": TEST_METADATA,
        "classes": CLASS_METADATA
    }

    if split not in metadata_paths:
        raise ValueError(
            "Invalid split. Choose train, validation, test, or classes."
        )

    file_path = metadata_paths[split]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {file_path}"
        )

    return pd.read_csv(file_path)


def load_image(image_path, image_size=(64, 64)):
    """
    Load an image, convert it to RGB, resize it,
    and normalize pixel values to [0, 1].
    """

    image_path = str(image_path)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(image_size)

        image_array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

    return image_array


def load_images_from_metadata(metadata, image_size=(64, 64)):
    images = []
    labels = []

    from src.config import RAW_DATA_DIR

    for _, row in metadata.iterrows():
        image_path = RAW_DATA_DIR / "GTSRB" / row["Path"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        images.append(load_image(image_path, image_size))
        labels.append(int(row["ClassId"]))

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int64)
    )