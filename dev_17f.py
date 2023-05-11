import tensorflow as tf
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping

from src.model import get_model
from src.utils import load_ds
from processing.grey_world.cc_layers import GreyEdge, GreyWorld, WhitePatch

# Model parameters
NUM_CLASSES = 17
MAX_EPOCHS = 300
REDUCE_LR_FACTOR = 0.2
REDUCE_LR_PATIENCE = 5
REDUCE_LR_MIN_LR = 1e-5
EARLY_STOP_PATIENCE = 25
LEARNING_RATE = 1e-5

# list of class labels
with open("17flowers_labels.txt", "r") as f:
    flower_labels = [line.strip() for line in f]

# Load dataset
train_ds, val_ds, test_ds = load_ds(cc=False)
# train_ds, val_ds, test_ds = load_ds(cc=True)  # If CC dataset

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR,
                              patience=REDUCE_LR_PATIENCE, min_lr=REDUCE_LR_MIN_LR)
early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
tensorboard = TensorBoard(
    log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

# CC Layers
grey_world_layer = GreyWorld()
white_patch_layer = WhitePatch()
grey_edge_layer = GreyEdge()

# Create the model
model = get_model(NUM_CLASSES)  # Also used with the CC'd FC4 dataset
# model = get_model(NUM_CLASSES, use_batchnorm=True)
# model = get_model(NUM_CLASSES, cc_layer=grey_world_layer)
# model = get_model(NUM_CLASSES, cc_layer=grey_edge_layer)
# model = get_model(NUM_CLASSES, cc_layer=white_patch_layer)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

# Fit
train_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MAX_EPOCHS,
    callbacks=[reduce_lr, early_stop, tensorboard],
    verbose=1
)

# Evaluate
score = model.evaluate(test_ds, verbose=1)
print(f'Test accuracy: {score[1]:.3f}')
