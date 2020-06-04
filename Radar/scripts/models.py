import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow.keras.initializers as tfki
import numpy as np


#####################################################################################################################
#####################################################################################################################


def convdlrm_init(H, W, C, Nout, nk, ks, lks, activ, init):
  inputs = tfk.Input(shape=[None, H, W, C])
  # Encoder
  conv = tfkl.TimeDistributed(tfkl.Conv2D(nk, ks, padding='same', 
                                          activation=activ, kernel_initializer=init))(inputs)
  conv = tfkl.TimeDistributed(tfkl.Conv2D(nk, ks, padding='same', 
                                          activation=activ, kernel_initializer=init))(conv)
  LN = tfkl.LayerNormalization()(conv)
  CL1, cl1_h, cl1_c, _, _ = tfkl.Bidirectional(tfkl.ConvLSTM2D(nk, ks, padding='same',
                                                               activation=activ, kernel_initializer=init,
                                                               return_sequences=True, return_state=True))(LN)
  CL2, cl2_h, cl2_c = tfkl.ConvLSTM2D(nk, ks, padding='same',
                                      activation=activ, kernel_initializer=init,
                                      return_state=True)(CL1)
  # Decoder 1
  input_dec = [CL2]
  for i in range(Nout-1):
    input_dec.append(tf.zeros_like(CL2))
  input_dec = tf.stack(input_dec, axis=1)
  CL3 = tfkl.ConvLSTM2D(nk, ks, padding='same',
                        activation=activ, kernel_initializer=init,
                        return_sequences=True)(input_dec, initial_state=[cl1_h, cl1_c])
  CL4 = tfkl.ConvLSTM2D(nk, ks, padding='same',
                        activation=activ, kernel_initializer=init,
                        return_sequences=True)(CL3, initial_state=[cl2_h, cl2_c])
  LN = tfkl.LayerNormalization()(CL4)
  # Deepen
  conv1 = tfkl.TimeDistributed(tfkl.Conv2D(nk, ks, padding='same',
                                           activation=activ, kernel_initializer=init))(LN)
  mp = tfkl.TimeDistributed(tfkl.MaxPooling2D())(conv1)
  conv2 = tfkl.TimeDistributed(tfkl.Conv2D(nk, ks, padding='same',
                                           activation=activ, kernel_initializer=init))(mp)
  mp = tfkl.TimeDistributed(tfkl.MaxPooling2D())(conv2)
  LN = tfkl.LayerNormalization()(mp)
  #
  conv3 = tfkl.TimeDistributed(tfkl.Conv2D(nk, ks, padding='same',
                                           activation=activ, kernel_initializer=init))(LN)
  mp = tfkl.TimeDistributed(tfkl.MaxPooling2D())(conv3)
  conv4 = tfkl.TimeDistributed(tfkl.Conv2D(nk, ks, padding='same',
                                           activation=activ, kernel_initializer=init))(mp)
  mp = tfkl.TimeDistributed(tfkl.MaxPooling2D())(conv4)
  LN = tfkl.LayerNormalization()(mp)
  #
  conv5 = tfkl.TimeDistributed(tfkl.Conv2D(nk, ks, padding='same',
                                           activation=activ, kernel_initializer=init))(LN)
  us = tfkl.TimeDistributed(tfkl.UpSampling2D())(conv5)
  concat = tfkl.Concatenate()([us, conv4])
  conv6 = tfkl.TimeDistributed(tfkl.Conv2D(nk, ks, padding='same',
                                           activation=activ, kernel_initializer=init))(concat)
  us = tfkl.TimeDistributed(tfkl.UpSampling2D())(conv6)
  concat = tfkl.Concatenate()([us, conv3])
  LN = tfkl.LayerNormalization()(concat)
  #
  conv7 = tfkl.TimeDistributed(tfkl.Conv2D(nk, ks, padding='same',
                                           activation=activ, kernel_initializer=init))(LN)
  us = tfkl.TimeDistributed(tfkl.UpSampling2D())(conv7)
  concat = tfkl.Concatenate()([us, conv2])
  conv8 = tfkl.TimeDistributed(tfkl.Conv2D(nk, ks, padding='same',
                                           activation=activ, kernel_initializer=init))(concat)
  us = tfkl.TimeDistributed(tfkl.UpSampling2D())(conv8)
  concat = tfkl.Concatenate()([us, conv1])
  LN = tfkl.LayerNormalization()(concat)
  # Decoder 2
  CL5 = tfkl.ConvLSTM2D(nk, ks, padding='same',
                        activation=activ, kernel_initializer=init,
                        return_sequences=True)(LN, initial_state=[cl1_h, cl1_c])
  CL6 = tfkl.ConvLSTM2D(nk, ks, padding='same',
                        activation=activ, kernel_initializer=init,
                        return_sequences=True)(CL5, initial_state=[cl2_h, cl2_c])
  LN = tfkl.LayerNormalization()(CL6)
  # Prediction
  preds = tfkl.Conv3D(1, lks, padding='same',
                      bias_initializer=tfki.Constant(value=-np.log(99)),
                      activation='sigmoid')(LN)
  return tfk.Model(inputs=inputs, outputs=preds)


#####################################################################################################################
#####################################################################################################################


def ddnet_init(H, W, C, Nout, nk, ks, lks, activ, init):
  # Inputs
  inputs_m = tfk.Input(shape=[None, H, W, C])
  inputs_c = tfk.Input(shape=[H, W, C])
  # Motion Encoder
  x = tfkl.TimeDistributed(tfkl.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init))(inputs_m)
  res1a = tfkl.Lambda(lambda x: x[:,-1])(x)
  x = tfkl.TimeDistributed(tfkl.MaxPooling2D())(x)
  x = tfkl.TimeDistributed(tfkl.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init))(x)
  res2a = tfkl.Lambda(lambda x: x[:,-1])(x)
  x = tfkl.TimeDistributed(tfkl.MaxPooling2D())(x)
  x = tfkl.LayerNormalization()(x)
  x = tfkl.ConvLSTM2D(nk, ks, padding='same', activation=activ, kernel_initializer=init, return_sequences=True)(x)
  ME = tfkl.ConvLSTM2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
  # Content Encoder
  x = tfkl.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(inputs_c)
  x = tfkl.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
  res1b = tfkl.BatchNormalization()(x)
  x = tfkl.MaxPooling2D()(res1b)
  x = tfkl.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
  x = tfkl.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
  res2b = tfkl.BatchNormalization()(x)
  x = tfkl.MaxPooling2D()(res2b)
  x = tfkl.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
  CE = tfkl.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
  # Combination layers
  x = tfkl.Concatenate()([CE, ME])
  x = tfkl.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
  x = tfkl.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
  x = tfkl.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
  x = tfkl.BatchNormalization()(x)
  # Decoder layers
  x = tfkl.UpSampling2D()(x)
  x = tfkl.Concatenate()([x, res2a, res2b])
  x = tfkl.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
  x = tfkl.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
  x = tfkl.BatchNormalization()(x)
  x = tfkl.UpSampling2D()(x)
  x = tfkl.Concatenate()([x, res1a, res1b])
  x = tfkl.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
  x = tfkl.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
  x = tfkl.BatchNormalization()(x)
  # Prediction
  input_p = [x]
  for i in range(Nout-1):
    input_p.append(tf.zeros_like(x))
  input_p = tf.stack(input_p, axis=1)
  x = tfkl.ConvLSTM2D(nk, ks, padding='same', activation=activ, kernel_initializer=init, return_sequences=True)(input_p)
  x = tfkl.ConvLSTM2D(nk, ks, padding='same', activation=activ, kernel_initializer=init, return_sequences=True)(x)
  x = tfkl.LayerNormalization()(x)
  preds = tfkl.Conv3D(1, lks, padding='same', activation='sigmoid', bias_initializer=tfki.Constant(value=-np.log(99)))(x)
  return tfk.Model(inputs=[inputs_m, inputs_c], outputs=preds)
