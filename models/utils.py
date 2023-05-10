import pathlib
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (RandomFlip, RandomRotation,
                                     RandomTranslation, RandomZoom)

# Constants
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def load_ds(cc:bool):
    """
        Return train/val/test dataset.

        `cc`: if True, will return CC'd dataset.
    """
    # Split directories
    path_to_data = "data/17flowers/"
    if cc:
        path_to_data = "data/17flowers/cc/"
    train_dir = pathlib.Path(path_to_data + "train")
    val_dir = pathlib.Path(path_to_data + "val")
    test_dir = pathlib.Path(path_to_data + "test")

    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
    )

    # Performance
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, test_ds


def get_data_augmentation_layer():
    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        RandomTranslation(0.2, 0.2),
        RandomRotation(0.2),
        RandomZoom(0.2),
    ])
    return data_augmentation


def run_experiment(model, train_ds, val_ds, test_ds, max_epochs:int, callbacks:list, n_trials=10):
    metrics = {
        "train_time": [],
        "test_time": [],
        "train_acc": [],
        "train_loss": [],
        "val_acc": [],
        "val_loss": [],
        "test_acc": [],
        "test_loss": [],
        "history": []
    }

    for i in range(n_trials):        
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

        start_time = time.perf_counter()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max_epochs,
            callbacks=callbacks,
            verbose=1)
        end_time = time.perf_counter()
        training_time = end_time - start_time

        start_time = time.perf_counter()
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        end_time = time.perf_counter()
        test_time = end_time - start_time

        metrics["train_time"].append(training_time)
        metrics["test_time"].append(test_time)
        metrics["train_acc"].append(history.history["accuracy"][-1])
        metrics["train_loss"].append(history.history["loss"][-1])
        metrics["val_acc"].append(history.history["val_accuracy"][-1])
        metrics["val_loss"].append(history.history["val_loss"][-1])
        metrics["test_acc"].append(test_acc)
        metrics["test_loss"].append(test_loss)
        metrics["history"].append(history)
    return metrics
