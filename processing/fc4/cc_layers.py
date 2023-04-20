import tensorflow as tf
from tensorflow import keras
from fcn import FCN
from config import *
from utils import get_session
from data_provider import load_data
import utils
from datasets import get_image_pack_fn
from data_provider import ImageRecord

from summary_utils import *


with tf.Graph().as_default():
    with get_session() as sess:
        fcn = FCN(sess=sess, name='../../pretrained_colorchecker/colorchecker_fold1and2.ckpt')
    # Create a variable scope to match the variable names in the checkpoint
    # with tf.compat.v1.variable_scope(""):
    #     # Restore the variables from the checkpoint
    #     saver = tf.compat.v1.train.import_meta_graph("../../pretrained_colorchecker/colorchecker_fold1and2.ckpt.meta")
    #     saver.restore(tf.compat.v1.get_default_session(), "../../pretrained_colorchecker/colorchecker_fold1and2.ckpt")

    # Save the restored graph as a SavedModel
        illums = tf.compat.v1.placeholder(tf.float32, shape=(None, 3), name='illums')
        images = tf.compat.v1.placeholder(tf.float32, shape=(None, 512, 512, 3), name='images')

        pixels = FCN.build_branches(images, 1.0)
        est = tf.reduce_sum(pixels, axis=(1, 2))
        merged = get_visualization(images, pixels, est, illums, (400, 400))
            
        tf.compat.v1.saved_model.simple_save(
            sess,
            "../../pretrained_colorchecker/v2_model",
            inputs={'image':images, 'illum':illums},  # Define input signatures
            outputs={'pixels':pixels, 'est':est, 'merged':merged}
        )

loaded_model = tf.saved_model.load("../../pretrained_colorchecker/v2_model")
print(loaded_model.signatures.keys())
# # Get the model's function signature (if multiple signatures are defined)
# signature = loaded_model.signatures["serving_default"]

# # Retrieve the input and output tensor names
# input_tensor_name = signature.inputs["input_tensor_name"].name
# output_tensor_name = signature.outputs["output_tensor_name"].name

# # Create a TensorFlow 2 compatible function
# @tf.function
# def predict(inputs):
#     # Call the loaded model's function using the input tensor name
#     outputs = signature(inputs={input_tensor_name: inputs})
#     return outputs[output_tensor_name]

# # Generate some input data
# inputs = ...

# # Call the predict function to get model predictions
# predictions = predict(inputs)
