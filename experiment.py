from datetime import datetime

import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from models.flowers import get_base_model, get_batchnorm_model, get_cc_model
from models.utils import load_ds, run_experiment
from processing.grey_world.cc_layers import GreyEdge, GreyWorld, WhitePatch

# Parameters
NUM_CLASSES = 17
MAX_EPOCHS = 300
REDUCE_LR_FACTOR = 0.2
REDUCE_LR_PATIENCE = 5
REDUCE_LR_MIN_LR = 1e-5
EARLY_STOP_PATIENCE = 25
N_TRIALS = 10

# CC layers
grey_world_layer = GreyWorld()
white_patch_layer = WhitePatch()
grey_edge_layer = GreyEdge()

# Create new models
model_base = get_base_model(NUM_CLASSES)
model_batch = get_batchnorm_model(NUM_CLASSES)
model_gw = get_cc_model(NUM_CLASSES, grey_world_layer)
model_ge = get_cc_model(NUM_CLASSES, grey_edge_layer)
model_wp = get_cc_model(NUM_CLASSES, white_patch_layer)
model_fc4 = get_base_model(NUM_CLASSES)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=REDUCE_LR_MIN_LR)
early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
callbacks = [reduce_lr, early_stop]

# Load datasets
train_ds, val_ds, test_ds = load_ds(cc=False)
train_ds_cc, val_ds_cc, test_ds_cc = load_ds(cc=True)

# Run experiments
metrics = {}
metrics["Base"] = run_experiment(model_base, train_ds, val_ds, test_ds, MAX_EPOCHS, callbacks, n_trials=N_TRIALS)
metrics["BatchNorm"] = run_experiment(model_batch, train_ds, val_ds, test_ds, MAX_EPOCHS, callbacks, n_trials=N_TRIALS)
metrics["GreyWorld"] = run_experiment(model_gw, train_ds, val_ds, test_ds, MAX_EPOCHS, callbacks, n_trials=N_TRIALS)
metrics["GreyEdge"] = run_experiment(model_ge, train_ds, val_ds, test_ds, MAX_EPOCHS, callbacks, n_trials=N_TRIALS)
metrics["WhitePatch"] = run_experiment(model_wp, train_ds, val_ds, test_ds, MAX_EPOCHS, callbacks, n_trials=N_TRIALS)
metrics["FC4"] = run_experiment(model_fc4, train_ds_cc, val_ds_cc, test_ds_cc, MAX_EPOCHS, callbacks, n_trials=N_TRIALS)

# Get the current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

# Export data to Excel sheet
dst_path = f"./out/{timestamp}_experiments_17flowers.xlsx"
with pd.ExcelWriter(dst_path, engine='xlsxwriter',) as writer:    
    end_data = pd.concat({k: pd.DataFrame(v) for k, v in metrics.items()}, axis=0, names=["Algorithm", "Trial"])
    end_data.drop("history", axis=1, inplace=True)
    end_data.to_excel(writer, "Final Data", merge_cells=False)

    for k, metric in metrics.items():
        histories = metric["history"]
        algo_data = pd.concat({f"{i}": pd.DataFrame(history.history) for i, history in enumerate(histories)}, axis=1)
        algo_data.to_excel(writer, f"{k} History", merge_cells=False)

print(f"Data saved to {dst_path}")