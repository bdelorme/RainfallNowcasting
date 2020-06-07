import os
FOLDERNAME = os.getcwd()+'/'
print('FOLDERNAME=' + FOLDERNAME)
#
import numpy as np
import tensorflow as tf
import csv
#
from pathlib import Path
import argparse
import json
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
################################## READ JSON EXP FILE ###################################
#########################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('param_file', type=Path)
args = parser.parse_args()
param = json.load(args.param_file.open())


#########################################################################################
################################## EXPERIMENT FEATURES ##################################
#########################################################################################
nom_test = param['nom_test']                        # NAME OF THE CURRENT TEST
archi = param['archi']                              # NETWORK ARCHITECTURE
new_size = param['new_size']                        # SIZE DATA
bs = param['batch_size']                            # BATCH SIZE
ep = param['nb_epochs']                             # NB EPOCHS


#########################################################################################
#################################### NETWORK FEATURES ###################################
#########################################################################################
loss = param['loss']
if loss=='masked_logcosh':
    loss=masked_logcosh
elif loss=='masked_mae':
    loss=masked_mae
elif loss=='masked_mse':
    loss=masked_mse
optimizer = tf.keras.optimizers.Adam(lr=param['learning_rate'])
nk = param['nb_filters']
ks = param['filter_size']
lks = param['last_filter_size']
activ = param['activation']
init = param['weights_initialization']


#########################################################################################
################################## CALLBACKS & METRICS ##################################
#########################################################################################
csv_logger = tf.keras.callbacks.CSVLogger('models/'+nom_test+'_train.csv')
term_nan = tf.keras.callbacks.TerminateOnNaN()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_masked_BMW', patience=5, mode='max', restore_best_weights=True)
callbacks_list = [term_nan, csv_logger, early_stopping]
#
metrics_list = [masked_acc, masked_ssim, masked_psnr, masked_cor, masked_prec, masked_recall, masked_BMW]


#########################################################################################
#################################### GLOBAL FEATURES ####################################
#########################################################################################
zone = "NW"                           # NW, SE
years = param['years']                # 2016, 2017, 2018
months = param['months']              # 1..12
parts_month = param['parts_month']    # 1,2,3 (each month is divided in 3 parts)
input_timeframes = param['input_timeframes']                   # how many timeframes for input
output_timeframes = param['output_timeframes']                 # how many timeframes for output
overlapping_data = param['overlapping']                        # data overlap in time (= 1) or not (= 0)
fraction_test = 0.1                   # fraction of test data
rainfall_threshold_value = 40.        # Value above which values are considered to be one
size_regions = param['size_regions']
threshold_rain_in_regions = param['threshold_rain_in_regions']


#########################################################################################
################################## ADDITIONAL FEATURES ##################################
#########################################################################################
features_bool = {'reflectivity': param['reflectivity_on'],
                 'rainfall quality': param['quality_on'],
                 'land sea': param['lsm_on'],
                 'elevation': param['relief_on']}
features_max_threshold = {'reflectivity': 60,
                          'rainfall quality': 100,
                          'land sea': 1,
                          'elevation': 629}
features_min_threshold = {'reflectivity': 0,
                          'rainfall quality': 0,
                          'land sea': 0,
                          'elevation': 0}
#
wf_model = 'arpege' # None for no model, otherwise arpege
weather_model_bool = {'temperature': param['temperature_on'],
                      'dew point temperature' : param['dew_temp_on'],
                      'humidity': param['humidity_on'],
                      'wind speed': param['wind_speed_on'],
                      'wind directions': param['wind_dir_on'],
                      'wind components': param['wind_comp_on'],
                      'pressure': param['pressure_on'],
                      'precipitation': param['precip_on']}
weather_model_max_threshold = {'temperature': 308,
                               'dew point temperature' : 308,
                               'humidity': 100,
                               'wind speed': 30,
                               'wind directions': 360,
                               'wind components': 30,
                               'pressure': 105000,
                               'precipitation': rainfall_threshold_value}
weather_model_min_threshold = {'temperature': 268,
                               'dew point temperature' : 268,
                               'humidity': 0,
                               'wind speed': 0,
                               'wind directions': 0,
                               'wind components': -30,
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
                                            rainfall_threshold_value,
                                            features_bool, weather_model_bool, wf_model,
                                            weather_model_max_threshold, weather_model_min_threshold,
                                            features_max_threshold, features_min_threshold,
                                            overlapping_data, threshold_rain_in_regions,
                                            size_regions)
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
model.save('models/'+nom_test+'.h5')
with open('models/'+nom_test+'_test.csv', 'w') as f:
    w = csv.DictWriter(f, results.keys())
    w.writeheader()
    w.writerow(results)



#########################################################################################
######################################### PLOT ##########################################
#########################################################################################
# 1- History
foldername = FOLDERNAME+'plots/'
plot_history(history, results, nom_test, save=True, foldername=foldername)

# 2- Train Example
itest = np.argmax(np.sum(X[:,:,:,:,0], axis=(1,2,3)))
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
           [size_regions, size_regions], input_timeframes, output_timeframes,
           lat, lon, nom_test, tag='train',
           save=True, foldername=foldername)

# 3- Test Example
itest = np.argmax(np.sum(X_test[:,:,:,:,0], axis=(1,2,3)))
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
           [size_regions, size_regions], input_timeframes, output_timeframes,
           lat, lon, nom_test, tag='test',
           save=True, foldername=foldername)

