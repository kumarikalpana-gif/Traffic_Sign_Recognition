
import tensorflow as tf

from src.config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS,
    NUM_CLASSES
)


def build_baseline_cnn(name="baseline_cnn"):
    """
    Recreate the original trained CNN architecture.
    """

    inputs = tf.keras.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    )

    # Convolutional block 1
    x = tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Convolutional block 2
    x = tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Convolutional block 3
    x = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Classification head
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=name
    )


def build_heavy_augmentation_cnn():
    """
    Same architecture as the baseline CNN.
    Heavy augmentation is applied separately during training.
    """

    return build_baseline_cnn(
        name="heavy_augmentation_cnn"
    )


def build_efficientnet_b0(pretrained=True, fine_tune=False):
    """
    Build EfficientNetB0 with a custom 43-class head.
    """

    inputs = tf.keras.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    )

    x = tf.keras.layers.Rescaling(255.0)(inputs)

    weights = "imagenet" if pretrained else None

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    )

    base_model.trainable = fine_tune

    if fine_tune:
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        for layer in base_model.layers[-30:]:
            if isinstance(
                layer,
                tf.keras.layers.BatchNormalization
            ):
                layer.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="efficientnet_b0"
    )
