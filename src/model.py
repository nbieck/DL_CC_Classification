from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D, ReLU, Rescaling, BatchNormalization, RandomFlip, RandomRotation,
                                     RandomTranslation, RandomZoom)

from src.utils import IMG_WIDTH, IMG_HEIGHT


def get_data_augmentation_layer():
    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical",
                   input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        RandomTranslation(0.2, 0.2),
        RandomRotation(0.2),
        RandomZoom(0.2),
    ])
    return data_augmentation


def get_model(num_classes: int, use_batchnorm: bool = False, cc_layer=None):
    model = Sequential()
    model.add(Rescaling(1./255))

    if cc_layer:
        model.add(cc_layer)

    model.add(get_data_augmentation_layer())
    model.add(Conv2D(32, 5, padding="same",
              input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))

    if use_batchnorm:
        model.add(BatchNormalization())

    model.add(ReLU())
    model.add(MaxPooling2D())
    model.add(Conv2D(32, 3, activation='relu', padding="same"))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, activation='relu', padding="same"))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, 3, activation='relu', padding="same"))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, 3, activation='relu', padding="same"))
    model.add(MaxPooling2D())
    model.add(Conv2D(512, 3, activation='relu', padding="same"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    return model
