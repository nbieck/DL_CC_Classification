import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import tensorflow_addons as tfa

def linearize_srgb(image: tf.Tensor) -> tf.Tensor:
    out = tf.identity(image)
    under_threshold = image <= 0.04045
    over_threshold = tf.logical_not(under_threshold)
    under_slice = tf.boolean_mask(out, under_threshold)
    under_slice /= 12.92
    over_slice = tf.boolean_mask(out, over_threshold)
    over_slice = tf.pow((over_slice + 0.055) / 1.055, 2.4)
    return out

def rgb_to_srgb(image: tf.Tensor) -> tf.Tensor:
    out = tf.identity(image)
    under_threshold = image <= 0.0031308
    over_threshold = tf.logical_not(under_threshold)
    
    under_slice = tf.boolean_mask(out, under_threshold)
    under_slice *= 12.92
    over_slice = tf.boolean_mask(out, over_threshold)
    over_slice = tf.pow(1.055 * over_slice, 1 / 2.4) - 0.055

    return out

def rescaleToPercentile(image: tf.Tensor, percentile: int) -> tf.Tensor:
    percentile = tf.math.reduce_max(tfp.stats.percentile(image, percentile, axis=(1,2), keepdims=True), axis=(1,2,3), keepdims=True)
    out = tf.identity(image)
    out /= percentile
    out = tf.clip_by_value(out, 0, 1)

    return out

class GreyWorld(keras.layers.Layer):
    """Assumes that layer input is a 4D tensor with dimensions [N,W,H,C], 
    N = Batch
    W = Width
    H = Height
    C = Channels"""

    def call(self, inputs):
        floatImage = linearize_srgb(inputs)

        estimated_illumination = tf.math.reduce_mean(floatImage, axis=(1,2), keepdims=True)
        floatImage /= estimated_illumination

        # we rescale such that the top five percent of pixel values are clipped (in one of the channels)
        floatImage = rescaleToPercentile(floatImage, 95)

        floatImage = rgb_to_srgb(floatImage)

        return inputs

class WhitePatch(keras.layers.Layer):

    def call(self, inputs):
        floatImage = linearize_srgb(inputs)

        estimated_illumination = tf.math.reduce_max(floatImage, axis=(1,2), keepdims=True)
        floatImage /= estimated_illumination

        floatImage = rescaleToPercentile(floatImage, 95)

        floatImage = rgb_to_srgb(floatImage)

        return floatImage

class GreyEdge(keras.layers.Layer):

    def call(self, inputs):
        floatImage = linearize_srgb(inputs)

        blurred = tfa.image.gaussian_filter2d(floatImage, filter_shape=19, sigma=3)
        dx, dy = tf.image.image_gradients(blurred)
        derivatives = tf.sqrt(dx * dx + dy * dy)

        estimated_illumination = tf.math.reduce_mean(derivatives, axis=(1,2), keepdims=True)
        floatImage /= estimated_illumination

        floatImage = rescaleToPercentile(floatImage, 95)

        floatImage = rgb_to_srgb(floatImage)

        return floatImage
