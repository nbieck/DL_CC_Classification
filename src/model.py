from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D, ReLU, Rescaling, BatchNormalization, RandomFlip, RandomRotation,
                                     RandomTranslation, RandomZoom)
import tensorflow as tf
from keras.applications import VGG16
from src.utils import IMG_WIDTH, IMG_HEIGHT, LEARNING_RATE
from keras import models
from keras import layers


def get_data_augmentation_layer():
    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical",
                   input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        RandomTranslation(0.2, 0.2),
        RandomRotation(0.2),
        RandomZoom(0.2),
    ])
    return data_augmentation


def get_model(num_classes: int, use_batchnorm: bool = False, cc_layer=None, learning_rate=1e-3):
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

    # Compile the model
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate),
        metrics=['accuracy'])
    return model


def get_vggmodel(num_classes: int, use_batchnorm: bool = False, cc_layer=None, learning_rate=1e-3):

    # Load the VGG model
    vgg_model = VGG16(weights='imagenet', include_top=False,
                     input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # freezes all layers in the VGG model except for the last four layers
    for layer in vgg_model.layers[:-4]:
        layer.trainable = False

    # Create the final model
    model = models.Sequential()

    model.add(Rescaling(1./255))

    if use_batchnorm:
        model.add(Conv2D(3, 5, padding="same",
                  input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
        model.add(BatchNormalization())
        model.add(ReLU())

    if cc_layer != None:
        # Add cc layers
        model.add(cc_layer)

    # Add the vgg model
    model.add(vgg_model)

    # Add new layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.RMSprop(
                      learning_rate=learning_rate),
                  metrics=['accuracy'])

    # Build the model
    model.build((None, IMG_HEIGHT, IMG_WIDTH, 3))

    return model
