import tensorflow as tf
import numpy as np


#####################################################################################################################
#####################################################################################################################


def convdlrm_init(H, W, C, Nout, nk, ks, lks, activ, init):
    inputs = tf.keras.Input(shape=[None, H, W, C])
    # Downsample
    conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(nk, ks, padding='same',
                                            activation=activ, kernel_initializer=init))(inputs)
    do1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.7))(conv1)
    conv2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(nk, ks, padding='same',
                                            activation=activ, kernel_initializer=init))(do1)
    mp1 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(conv2)
    LN1 = tf.keras.layers.LayerNormalization()(mp1)
    do2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.7))(LN1)
    # Encoder
    CL1, cl1_h, cl1_c = tf.keras.layers.ConvLSTM2D(nk, ks, padding='same',
                                                   activation=activ, kernel_initializer=init,
                                                   return_state=True)(do2)
    # Decoder
    do3 = tf.keras.layers.Dropout(0.7)(CL1)
    input_dec = tf.stack([do3, tf.zeros_like(do3), tf.zeros_like(do3), tf.zeros_like(do3), tf.zeros_like(do3)], axis=1)
    CL2 = tf.keras.layers.ConvLSTM2D(nk, ks, padding='same',
                                     activation=activ, kernel_initializer=init,
                                     return_sequences=True)(input_dec,
                                                            initial_state=[cl1_h, cl1_c])
    LN2 = tf.keras.layers.LayerNormalization()(CL2)
    do4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.7))(LN2)
    # Upsample
    conv3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(nk, ks, padding='same',
                                            activation=activ, kernel_initializer=init))(do4)
    do5 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.7))(conv3)
    conv4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(nk, ks, padding='same',
                                        activation=activ, kernel_initializer=init))(do5)
    us1 = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D())(conv4)
    LN3 = tf.keras.layers.LayerNormalization()(us1)
    # Prediction
    do6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.7))(LN3)
    preds = tf.keras.layers.Conv3D(1, lks, padding='same',
                                   bias_initializer=tf.keras.initializers.Constant(value=-np.log(99)),
                                   activation='sigmoid')(do6)
    return tf.keras.Model(inputs=inputs, outputs=preds)


#####################################################################################################################
#####################################################################################################################


def ddnet_init(H, W, C, Nout, nk, ks, lks, activ, init):
    # Inputs
    input_m = tf.keras.Input(shape=[None, H, W, C])
    input_c = tf.keras.Input(shape=[H, W, C])
    # Motion Encoder
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init))(input_m)
    res1a = tf.keras.layers.Lambda(lambda x: x[:,-1])(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init))(x)
    res2a = tf.keras.layers.Lambda(lambda x: x[:,-1])(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2))(x)
    ME = tf.keras.layers.ConvLSTM2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
    # Content Encoder
    x = tf.keras.layers.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(input_c)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
    res1b = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(res1b)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
    res2b = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(res2b)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    CE = tf.keras.layers.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
    # Decoder
    x = tf.keras.layers.Concatenate()([CE, ME])
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Concatenate()([x, res2a, res2b])
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Concatenate()([x, res1a, res1b])
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(nk, ks, padding='same', activation=activ, kernel_initializer=init)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    # Prediction
    x = tf.stack([x, tf.zeros_like(x), tf.zeros_like(x), tf.zeros_like(x), tf.zeros_like(x)], axis=1)
    x = tf.keras.layers.ConvLSTM2D(nk, ks, padding='same', activation=activ, kernel_initializer=init, return_sequences=True)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2))(x)
    preds = tf.keras.layers.Conv3D(1, lks, padding='same', activation='sigmoid',
                                   bias_initializer=tf.keras.initializers.Constant(value=-np.log(99)))(x)
    return tf.keras.Model(inputs=[input_m, input_c], outputs=preds)

