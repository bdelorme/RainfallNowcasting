import tensorflow as tf
import tensorflow_probability as tfp


def cor(y_gt, y_pred):
  return tfp.stats.correlation(y_gt, y_pred, sample_axis=None, event_axis=None)


def ssim(y_gt, y_pred, max_val=1.):
  return tf.image.ssim(y_gt, y_pred, max_val)


def psnr(y_gt, y_pred, max_val=1.):
  return tf.image.psnr(y_gt, y_pred, max_val)

def masked_logcosh(y_true, y_pred):
  mask = tf.cast(tf.greater_equal(y_true, 0), tf.float32)
  loss = tf.keras.losses.logcosh(tf.multiply(y_true, mask), tf.multiply(y_pred, mask))
  return loss

def masked_mae(y_true, y_pred):
  mask = tf.cast(tf.greater_equal(y_true, 0), tf.float32)
  loss = tf.keras.losses.mae(tf.multiply(y_true, mask), tf.multiply(y_pred, mask))
  return loss

def masked_mse(y_true, y_pred):
  mask = tf.cast(tf.greater_equal(y_true, 0), tf.float32)
  loss = tf.keras.losses.mse(tf.multiply(y_true, mask), tf.multiply(y_pred, mask))
  return loss