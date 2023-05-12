import pathlib
import time

import tensorflow as tf
import pandas as pd
import logging

# Constants
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
LEARNING_RATE = 1e-3


def load_ds(ds_name: str, cc: bool):
    """
        Return train/val/test dataset.

        `ds_name`: either '17flowers' or '102flowers'.

        `cc`: if True, will return CC'd dataset.
    """

    # Split directories
    path_to_data = f"data/{ds_name}/"
    if cc:
        path_to_data = f"data/{ds_name}/cc/"
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


def run_experiment(get_model, train_ds, val_ds, test_ds, n_epochs: int, callbacks: list, n_trials: int = 10, **kwargs):
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
        logging.info(f"Running trial {i+1}/{n_trials}.")

        model = get_model(**kwargs)

        start_time = time.perf_counter()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=n_epochs,
            callbacks=callbacks,
            verbose=1)
        end_time = time.perf_counter()
        training_time = end_time - start_time

        start_time = time.perf_counter()
        test_loss, test_acc = model.evaluate(test_ds, verbose=1)
        end_time = time.perf_counter()
        test_time = end_time - start_time

        train_acc = history.history["accuracy"][-1]
        val_acc = history.history["val_accuracy"][-1]

        metrics["train_time"].append(training_time)
        metrics["test_time"].append(test_time)
        metrics["train_acc"].append(train_acc)
        metrics["train_loss"].append(history.history["loss"][-1])
        metrics["val_acc"].append(val_acc)
        metrics["val_loss"].append(history.history["val_loss"][-1])
        metrics["test_acc"].append(test_acc)
        metrics["test_loss"].append(test_loss)
        metrics["history"].append(history)

        # Log information
        logging.info(
            f"Trial {i+1}/{n_trials}: train_acc {train_acc} | val_acc {val_acc} | test_acc {test_acc}")

        # Clear Keras backend session to free GPU memory
        tf.keras.backend.clear_session()
    return metrics


def write_to_excel(dst_path, metrics):
    with pd.ExcelWriter(dst_path, engine='xlsxwriter',) as writer:
        end_data = pd.concat({k: pd.DataFrame(
            v) for k, v in metrics.items()}, axis=0, names=["Algorithm", "Trial"])
        end_data.drop("history", axis=1, inplace=True)
        end_data.to_excel(writer, "Final Data", merge_cells=False)

        for k, metric in metrics.items():
            histories = metric["history"]
            algo_data = pd.concat({f"{i}": pd.DataFrame(history.history)
                                   for i, history in enumerate(histories)}, axis=1)
            algo_data.to_excel(writer, f"{k} History", merge_cells=False)


def convert_seconds(seconds):
    hours, remainder = divmod(seconds, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)  # 60 seconds in a minute
    return hours, minutes
