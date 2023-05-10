from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D, ReLU, Rescaling, BatchNormalization)

from models.utils import get_data_augmentation_layer, IMG_WIDTH, IMG_HEIGHT


def get_base_model(num_classes):
  model = Sequential([
    Rescaling(1./255),
    get_data_augmentation_layer(),
    Conv2D(32, 5, activation='relu', padding="same", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, activation='relu', padding="same"),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu', padding="same"),
    MaxPooling2D(),
    Conv2D(128, 3, activation='relu', padding="same"),
    MaxPooling2D(),
    Conv2D(256, 3, activation='relu', padding="same"),
    MaxPooling2D(),
    Conv2D(512, 3, activation='relu', padding="same"),
    MaxPooling2D(),
    Flatten(),
    ReLU(),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
  ])
  return model


def get_batchnorm_model(num_classes):
  model = Sequential([
    Rescaling(1./255),
    get_data_augmentation_layer(),
    Conv2D(32, 5, padding="same", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(),
    Conv2D(32, 3, activation='relu', padding="same"),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu', padding="same"),
    MaxPooling2D(),
    Conv2D(128, 3, activation='relu', padding="same"),
    MaxPooling2D(),
    Conv2D(256, 3, activation='relu', padding="same"),
    MaxPooling2D(),
    Conv2D(512, 3, activation='relu', padding="same"),
    MaxPooling2D(),
    Flatten(),
    ReLU(),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
  ])
  return model


def get_cc_model(num_classes, cc_layer):
  model = Sequential([
    Rescaling(1./255),
    cc_layer,
    get_data_augmentation_layer(),
    Conv2D(32, 5, activation='relu', padding="same", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, activation='relu', padding="same"),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu', padding="same"),
    MaxPooling2D(),
    Conv2D(128, 3, activation='relu', padding="same"),
    MaxPooling2D(),
    Conv2D(256, 3, activation='relu', padding="same"),
    MaxPooling2D(),
    Conv2D(512, 3, activation='relu', padding="same"),
    MaxPooling2D(),
    Flatten(),
    ReLU(),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
  ])
  return model