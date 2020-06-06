import tensorflow as tf
import tensorflow_probability as tfp


def cor(y_true, y_pred):
  return tfp.stats.correlation(y_true, y_pred, sample_axis=None, event_axis=None)

def masked_cor(y_true, y_pred):
  mask = tf.cast(tf.greater_equal(y_true, 0), tf.float32)
  return tfp.stats.correlation(tf.multiply(y_true, mask), tf.multiply(y_pred, mask), sample_axis=None, event_axis=None)


def ssim(y_true, y_pred, max_val=1.):
  return tf.image.ssim(y_true, y_pred, max_val)

def masked_ssim(y_true, y_pred, max_val=1.):
  mask = tf.cast(tf.greater_equal(y_true, 0), tf.float32)
  return tf.image.ssim(tf.multiply(y_true, mask), tf.multiply(y_pred, mask), max_val)


def psnr(y_true, y_pred, max_val=1.):
  return tf.image.psnr(y_true, y_pred, max_val)

def masked_psnr(y_true, y_pred, max_val=1.):
  mask = tf.cast(tf.greater_equal(y_true, 0), tf.float32)
  return tf.image.psnr(tf.multiply(y_true, mask), tf.multiply(y_pred, mask), max_val)


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


def prec(y_true, y_pred, treshold=1./40.):
    y_true_pos = tf.cast(tf.greater_equal(y_true, treshold), tf.float32)
    y_pred_pos = tf.cast(tf.greater_equal(y_pred, treshold), tf.float32)
    TP = tf.math.count_nonzero(y_true_pos * y_pred_pos)
    FP = tf.math.count_nonzero(y_pred_pos * (y_true_pos - 1))
    precision = TP / (TP + FP)
    return precision

def masked_prec(y_true, y_pred, treshold=1./40.):
    mask = tf.cast(tf.greater_equal(y_true, 0), tf.float32)
    y_true_pos = tf.cast(tf.greater_equal(y_true*mask, treshold), tf.float32)
    y_pred_pos = tf.cast(tf.greater_equal(y_pred*mask, treshold), tf.float32)
    TP = tf.math.count_nonzero(y_true_pos * y_pred_pos)
    FP = tf.math.count_nonzero(y_pred_pos * (y_true_pos - 1))
    precision = TP / (TP + FP)
    return precision


def recall(y_true, y_pred, treshold=1./40.):
    y_true_pos = tf.cast(tf.greater_equal(y_true, treshold), tf.float32)
    y_pred_pos = tf.cast(tf.greater_equal(y_pred, treshold), tf.float32)
    TP = tf.math.count_nonzero(y_true_pos * y_pred_pos)
    FN = tf.math.count_nonzero((y_pred_pos - 1) * y_true_pos)
    recall = TP / (TP + FN)
    return recall

def masked_recall(y_true, y_pred, treshold=1./40.):
    mask = tf.cast(tf.greater_equal(y_true, 0), tf.float32)
    y_true_pos = tf.cast(tf.greater_equal(y_true*mask, treshold), tf.float32)
    y_pred_pos = tf.cast(tf.greater_equal(y_pred*mask, treshold), tf.float32)
    TP = tf.math.count_nonzero(y_true_pos * y_pred_pos)
    FN = tf.math.count_nonzero((y_pred_pos - 1) * y_true_pos)
    recall = TP / (TP + FN)
    return recall


def acc(y_true, y_pred, treshold=1./40.):
    y_true_pos = tf.cast(tf.greater_equal(y_true, treshold), tf.float32)
    y_pred_pos = tf.cast(tf.greater_equal(y_pred, treshold), tf.float32)
    TP = tf.math.count_nonzero(y_true_pos * y_pred_pos)
    TN = tf.math.count_nonzero((y_pred_pos - 1) * (y_true_pos - 1))
    FP = tf.math.count_nonzero(y_pred_pos * (y_true_pos - 1))
    FN = tf.math.count_nonzero((y_pred_pos - 1) * y_true_pos)
    acc = (TP+TN) / (TP+FN+TN+FP)
    return acc

def masked_acc(y_true, y_pred, treshold=1./40.):
    mask = tf.cast(tf.greater_equal(y_true, 0), tf.float32)
    y_true_pos = tf.cast(tf.greater_equal(y_true*mask, treshold), tf.float32)
    y_pred_pos = tf.cast(tf.greater_equal(y_pred*mask, treshold), tf.float32)
    TP = tf.math.count_nonzero(y_true_pos * y_pred_pos)
    TN = tf.math.count_nonzero((y_pred_pos - 1) * (y_true_pos - 1))
    FP = tf.math.count_nonzero(y_pred_pos * (y_true_pos - 1))
    FN = tf.math.count_nonzero((y_pred_pos - 1) * y_true_pos)
    acc = (TP+TN) / (TP+FN+TN+FP)
    return acc


def BMW(y_true, y_pred, treshold=1./40.):
    BMW = ( tf.cast(cor(y_true, y_pred), tf.float32) + tf.cast(prec(y_true, y_pred, treshold), tf.float32) + tf.cast(recall(y_true, y_pred, treshold), tf.float32) ) / 3.
    return BMW

def masked_BMW(y_true, y_pred, treshold=1./40.):
    BMW = ( tf.cast(masked_cor(y_true, y_pred), tf.float32) + tf.cast(masked_prec(y_true, y_pred, treshold), tf.float32) + tf.cast(masked_recall(y_true, y_pred, treshold), tf.float32) ) / 3.
    return BMW

