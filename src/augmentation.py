
import tensorflow as tf


def get_standard_augmentation():
    """
    Return the standard augmentation pipeline used
    for the baseline CNN and EfficientNetB0.
    """

    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomContrast(0.10),
        tf.keras.layers.RandomBrightness(0.10),
    ], name="standard_augmentation")


def get_heavy_augmentation():
    """
    Return the stronger augmentation pipeline used
    for the heavy-augmentation CNN.
    """

    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomTranslation(0.10, 0.10),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.25),
        tf.keras.layers.RandomBrightness(0.20),
    ], name="heavy_augmentation")
