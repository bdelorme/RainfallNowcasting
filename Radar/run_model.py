import os
FOLDERNAME = os.getcwd()+'/'
print('FOLDERNAME =' + FOLDERNAME)
#
import numpy as np
import tensorflow as tf
import csv
#
import sys
sys.path.append(FOLDERNAME+'scripts/')
#
from data_lab import *
from utils import *
from models import *
from metrics import *
from plot_tools import *


#########################################################################################
################################## EXPERIMENT FEATURES ##################################
#########################################################################################
nom_test = 'rainfall_only'      # NAME OF THE CURRENT TEST
archi = 'ddnet'                 # NETWORK ARCHITECTURE
new_size = [64,64]              # SIZE DATA (None = keep initial size)
bs = 10                         # BATCH SIZE
ep = 2                          # NB EPOCHS


#########################################################################################
#################################### NETWORK FEATURES ###################################
#########################################################################################
loss = 'logcosh'
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
nk = 128
ks = 5
lks = 3
if archi=='convdlrm':
    activ = 'selu'
    init = 'lecun_normal'
elif archi=='ddnet':
    activ = 'relu'
    init = 'he_normal'
    
    
#########################################################################################
################################## CALLBACKS & METRICS ##################################
#########################################################################################
csv_logger = tf.keras.callbacks.CSVLogger('models/'+archi+'_'+nom_test+'_train.csv')
term_nan = tf.keras.callbacks.TerminateOnNaN()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
callbacks_list = [term_nan, csv_logger, early_stopping]
#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)
#callbacks_list = [term_nan, reduce_lr, csv_logger, early_stopping]
#
metrics_list = ['acc', ssim, psnr, cor,
                tf.keras.metrics.Precision(name='prec'),
                tf.keras.metrics.Recall(name='recall')]   

    
#########################################################################################
#################################### GLOBAL FEATURES ####################################
#########################################################################################
zone = "NW"                           # NW, SE
years = [2017]                        # 2016, 2017, 2018
months = [8]                          # 1...12
parts_month = [3]                     # 1,2,3 (each month is divided in 3 parts)
input_timeframes = 10                 # how many timeframes for input
output_timeframes = 5                 # how many timeframes for output
overlapping_data = 0                  # data overlap in time (= 1) or not (= 0)
fraction_test = 0.1                   # fraction of test data
rainfall_threshold_value = 80         # Value above which values are considered to be one


#########################################################################################
################################## ADDITIONAL FEATURES ##################################
#########################################################################################
features_bool = {'reflectivity': 0, 
                 'rainfall quality': 0,
                 'land sea': 0, 
                 'elevation': 0}
features_max_threshold = {'reflectivity': 60, 
                          'rainfall quality': 100, 
                          'land sea': 1, 
                          'elevation': 629}
features_min_threshold = {'reflectivity': 0, 
                          'rainfall quality': 0, 
                          'land sea': 0, 
                          'elevation': 0}
#
wf_model = None # None for no model, otherwise arpege
weather_model_bool = {'temperature': 1, 
                      'dew point temperature' : 1,
                      'humidity': 1, 
                      'wind speed': 1, 
                      'wind directions': 1,
                      'wind components': 1, 
                      'pressure': 1,
                      'precipitation': 1}
weather_model_max_threshold = {'temperature': 313, 
                               'dew point temperature' : 313,
                               'humidity': 100, 
                               'wind speed': 35, 
                               'wind directions': 360,
                               'wind components': 35, 
                               'pressure': 105000,
                               'precipitation': rainfall_threshold_value}
weather_model_min_threshold = {'temperature': 263, 
                               'dew point temperature' : 263,
                               'humidity': 0, 
                               'wind speed': 0, 
                               'wind directions': 0,
                               'wind components': -35, 
                               'pressure': 96000,
                               'precipitation': 0}
#########################################################################################
#########################################################################################
#########################################################################################


#########################################################################################
####################################### MAKE DATA #######################################
#########################################################################################
data_dir = FOLDERNAME+'MeteoNet/'
X, y, features_list = get_data_and_features(data_dir, zone, years, months, parts_month, 
                                            new_size, input_timeframes, output_timeframes,
                                            fraction_test, rainfall_threshold_value, 
                                            features_bool, weather_model_bool, wf_model,
                                            weather_model_max_threshold, weather_model_min_threshold,
                                            features_max_threshold, features_min_threshold,
                                            overlapping_data)
print('X shape = ({0}, {1}, {2}, {3}, {4})'.format(X.shape[0],X.shape[1],X.shape[2],X.shape[3],X.shape[4]))
print('y shape = ({0}, {1}, {2}, {3}, {4})'.format(y.shape[0],y.shape[1],y.shape[2],y.shape[3],y.shape[4]))
print('Additional features: [%s]' % ', '.join(features_list))
X, y, X_test, y_test = split_train_test(fraction_test, X, y)
print('X_train shape = ({0}, {1}, {2}, {3}, {4})'.format(X.shape[0],X.shape[1],X.shape[2],X.shape[3],X.shape[4]))
print('y_train shape = ({0}, {1}, {2}, {3}, {4})'.format(y.shape[0],y.shape[1],y.shape[2],y.shape[3],y.shape[4]))
print('X_test shape = ({0}, {1}, {2}, {3}, {4})'.format(X_test.shape[0],X_test.shape[1],X_test.shape[2],X_test.shape[3],X_test.shape[4]))
print('y_test shape = ({0}, {1}, {2}, {3}, {4})'.format(y_test.shape[0],y_test.shape[1],y_test.shape[2],y_test.shape[3],y_test.shape[4]))
if archi=='ddnet':
    X_content = get_content_data(X)
    X_content_test = get_content_data(X_test)
N, T, H, W, C = X.shape




#########################################################################################
###################################### INIT MODEL #######################################
#########################################################################################
if archi=='convdlrm':
    model = convdlrm_init(H, W, C, output_timeframes, nk, ks, lks, activ, init)
elif archi=='ddnet':
    model = ddnet_init(H, W, C, output_timeframes, nk, ks, lks, activ, init)
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics_list)

#########################################################################################
####################################### RUN MODEL #######################################
#########################################################################################
if archi=='convdlrm':
    history = model.fit(X, y, 
                        batch_size=bs, 
                        epochs=ep,
                        callbacks=callbacks_list,
                        validation_split=0.1)
    results = model.evaluate(X_test, y_test, batch_size=bs, return_dict=True)
elif archi=='ddnet':
    history = model.fit([X, X_content], y, 
                        batch_size=bs, 
                        epochs=ep,
                        callbacks=callbacks_list,
                        validation_split=0.1)
    results = model.evaluate([X_test, X_content_test], y_test, batch_size=bs, return_dict=True)


#########################################################################################
###################################### SAVE MODEL #######################################
#########################################################################################
model.save('models/'+archi+'_'+nom_test+'.h5')
with open('models/'+archi+'_'+nom_test+'_test.csv', 'w') as f:
    w = csv.DictWriter(f, results.keys())
    w.writeheader()
    w.writerow(results)



#########################################################################################
######################################### PLOT ##########################################
#########################################################################################
# 1- History
foldername = FOLDERNAME+'plots/'
plot_history(history, results, archi, nom_test, save=True, foldername=foldername)

# 2- Train Example
itest = 3
track = tf.expand_dims(X[itest,:,:,:,0], axis=-1)
true_track = np.concatenate((track, y[itest]), axis=0)
if archi == 'convdlrm':
    track = np.concatenate((track[None,:,:,:,:], model.predict(X[itest][None,:,:,:,:])), axis=1)
elif archi == 'ddnet':
    track_m = X[itest]
    track_c = X_content[itest]
    track = np.concatenate((track[None,:,:,:,:], model.predict([track_m[None,:,:,:,:], track_c[None,:,:,:]])), axis=1)
lat, lon = get_coords(data_dir, zone)
plot_track(true_track, track, rainfall_threshold_value, 
           new_size, input_timeframes, output_timeframes, 
           lat, lon, archi, nom_test, tag='train', 
           save=True, foldername=foldername)

# 3- Test Example
itest = 3
track = tf.expand_dims(X_test[itest,:,:,:,0], axis=-1)
true_track = np.concatenate((track, y_test[itest]), axis=0)
if archi == 'convdlrm':
    track = np.concatenate((track[None,:,:,:,:], model.predict(X_test[itest][None,:,:,:,:])), axis=1)
elif archi == 'ddnet':
    track_m = X_test[itest]
    track_c = X_content_test[itest]
    track = np.concatenate((track[None,:,:,:,:], model.predict([track_m[None,:,:,:,:], track_c[None,:,:,:]])), axis=1)
lat, lon = get_coords(data_dir, zone)
plot_track(true_track, track, rainfall_threshold_value, 
           new_size, input_timeframes, output_timeframes, 
           lat, lon, archi, nom_test, tag='train', 
           save=True, foldername=foldername)

