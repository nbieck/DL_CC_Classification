from datetime import datetime

from processing.grey_world.cc_layers import GreyEdge, GreyWorld, WhitePatch
from src.model import get_model, get_vggmodel
from src.utils import load_ds, run_experiment, write_to_excel, convert_seconds
import logging
import time


# Parameters
PARAMS_17F = {
    "name": "17flowers",
    "n_classes": 17,
    "n_epochs": 50
}
PARAMS_102F = {
    "name": "102flowers",
    "n_classes": 102,
    "n_epochs": 100
}
LEARNING_RATE = 1e-3
N_TRIALS = 10

# Configure the logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# CC layers
grey_world_layer = GreyWorld()
white_patch_layer = WhitePatch()
grey_edge_layer = GreyEdge()

# Get start time
start_time = time.time()

for model in [get_model, get_vggmodel]:
    for params in [PARAMS_17F, PARAMS_102F]:
        # Load datasets
        train_ds, val_ds, test_ds = load_ds(ds_name=params["name"], cc=False)
        train_ds_cc, val_ds_cc, test_ds_cc = load_ds(
            ds_name=params["name"], cc=True)

        # Run experiments
        metrics = {}
        metrics["Base"] = run_experiment(
            model, train_ds, val_ds, test_ds, n_epochs=params["n_epochs"], learning_rate=LEARNING_RATE, callbacks=None, n_trials=N_TRIALS, num_classes=params["n_classes"])
        metrics["BatchNorm"] = run_experiment(model, train_ds, val_ds, test_ds, n_epochs=params["n_epochs"], learning_rate=LEARNING_RATE,
                                            callbacks=None, n_trials=N_TRIALS, num_classes=params["n_classes"], use_batchnorm=True)
        metrics["GreyWorld"] = run_experiment(model, train_ds, val_ds, test_ds, n_epochs=params["n_epochs"], learning_rate=LEARNING_RATE,
                                            callbacks=None, n_trials=N_TRIALS, num_classes=params["n_classes"], cc_layer=grey_world_layer)
        metrics["GreyEdge"] = run_experiment(model, train_ds, val_ds, test_ds, n_epochs=params["n_epochs"], learning_rate=LEARNING_RATE,
                                            callbacks=None, n_trials=N_TRIALS, num_classes=params["n_classes"], cc_layer=grey_edge_layer)
        metrics["WhitePatch"] = run_experiment(model, train_ds, val_ds, test_ds, n_epochs=params["n_epochs"], learning_rate=LEARNING_RATE,
                                            callbacks=None, n_trials=N_TRIALS, num_classes=params["n_classes"], cc_layer=white_patch_layer)
        metrics["FC4"] = run_experiment(model, train_ds, val_ds, test_ds, n_epochs=params["n_epochs"],
                                        learning_rate=LEARNING_RATE, callbacks=None, n_trials=N_TRIALS, num_classes=params["n_classes"])

        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

        # Export data to Excel sheet
        dst_path = f"./out/{timestamp}_experiments_{params['name']}.xlsx"
        write_to_excel(dst_path, metrics)

        logging.info(f"Data for {params['name']} saved to {dst_path}.")

# Get end time and calculate total time spent
end_time = time.time()
total_time = end_time - start_time

# Convert to hours and minutes
hours, minutes = convert_seconds(total_time)

# Log some messages
logging.info(f'Finished! The experiment took {hours}h and {minutes}min.')
