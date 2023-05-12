from datetime import datetime

from processing.grey_world.cc_layers import GreyEdge, GreyWorld, WhitePatch
from src.model import get_model, get_vggmodel
from src.utils import load_ds, run_experiment, write_to_excel, convert_seconds, funy_log, FUNY_LOG_LEVEL
import logging
import time
import os

# Parameters
PARAMS_17F = {
    "name": "17flowers",
    "n_classes": 17,
    "n_epochs": 100
}
PARAMS_102F = {
    "name": "102flowers",
    "n_classes": 102,
    "n_epochs": 200
}
LEARNING_RATE = 1e-3
N_TRIALS = 10
# Change this to get_vggmodel to run the experiment on VGG
GET_MODEL_FUNCTION = get_model

# Configure the logging
logging.addLevelName(FUNY_LOG_LEVEL, "FUNY")
logging.Logger.funy = funy_log
logging.basicConfig(level=FUNY_LOG_LEVEL,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CC layers
grey_world_layer = GreyWorld()
white_patch_layer = WhitePatch()
grey_edge_layer = GreyEdge()

# Make the out directory
os.makedirs("out", exist_ok=True)

# Get start time
start_time = time.time()

for params in [PARAMS_17F, PARAMS_102F]:
    logger.funy(f"Loading dataset {params['name']}.")

    # Load datasets
    train_ds, val_ds, test_ds = load_ds(ds_name=params["name"], cc=False)
    train_ds_cc, val_ds_cc, test_ds_cc = load_ds(
        ds_name=params["name"], cc=True)

    # Run experiments
    metrics = {}

    logger.funy(f"Starting experiment: {params['name']} - Base.")

    metrics["Base"] = run_experiment(
        GET_MODEL_FUNCTION, train_ds, val_ds, test_ds, n_epochs=params["n_epochs"], learning_rate=LEARNING_RATE, callbacks=None, n_trials=N_TRIALS, num_classes=params["n_classes"])

    logger.funy(f"Starting experiment: {params['name']} - BatchNorm.")

    metrics["BatchNorm"] = run_experiment(GET_MODEL_FUNCTION, train_ds, val_ds, test_ds, n_epochs=params["n_epochs"], learning_rate=LEARNING_RATE,
                                          callbacks=None, n_trials=N_TRIALS, num_classes=params["n_classes"], use_batchnorm=True)

    logger.funy(f"Starting experiment: {params['name']} - GreyWorld.")

    metrics["GreyWorld"] = run_experiment(GET_MODEL_FUNCTION, train_ds, val_ds, test_ds, n_epochs=params["n_epochs"], learning_rate=LEARNING_RATE,
                                          callbacks=None, n_trials=N_TRIALS, num_classes=params["n_classes"], cc_layer=grey_world_layer)

    logger.funy(f"Starting experiment: {params['name']} - GreyEdge.")

    metrics["GreyEdge"] = run_experiment(GET_MODEL_FUNCTION, train_ds, val_ds, test_ds, n_epochs=params["n_epochs"], learning_rate=LEARNING_RATE,
                                         callbacks=None, n_trials=N_TRIALS, num_classes=params["n_classes"], cc_layer=grey_edge_layer)

    logger.funy(f"Starting experiment: {params['name']} - WhitePatch.")

    metrics["WhitePatch"] = run_experiment(GET_MODEL_FUNCTION, train_ds, val_ds, test_ds, n_epochs=params["n_epochs"], learning_rate=LEARNING_RATE,
                                           callbacks=None, n_trials=N_TRIALS, num_classes=params["n_classes"], cc_layer=white_patch_layer)

    logger.funy(f"Starting experiment: {params['name']} - FC4.")

    metrics["FC4"] = run_experiment(GET_MODEL_FUNCTION, train_ds_cc, val_ds_cc, test_ds_cc, n_epochs=params["n_epochs"],
                                    learning_rate=LEARNING_RATE, callbacks=None, n_trials=N_TRIALS, num_classes=params["n_classes"])

    logger.funy(
        f"All experiments for {params['name']} finished. Saving data...")

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    # Export data to Excel sheet
    dst_path = f"./out/{timestamp}_experiments_{params['name']}.xlsx"
    write_to_excel(dst_path, metrics)

    logger.funy(f"Data for {params['name']} saved to {dst_path}.")

# Get end time and calculate total time spent
end_time = time.time()
total_time = end_time - start_time

# Convert to hours and minutes
hours, minutes = convert_seconds(total_time)

# Log some messages
logger.funy(f'Finished! The experiment took {hours}h and {minutes}min.')
