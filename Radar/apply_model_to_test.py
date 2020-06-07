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
import sys
sys.path.append(FOLDERNAME+'scripts/')
#
from data_lab import *
from utils import *
from models import *
from metrics import *
from plot_tools import *


### USAGE: python apply_model_to_new_test.py --nargs ddnet_rainonly.json 13 2018 3
# with 13=nb of tests to do
# 2018: year not used for train
# 3: month not used for train


### READ INPUTS
parser = argparse.ArgumentParser()
parser.add_argument('--nargs', nargs='+')
args = parser.parse_args()
param = json.load(Path(args.nargs[0]).open())


#########################################################################################
################################## EXPERIMENT FEATURES ##################################
#########################################################################################
nb_test = int(args.nargs[1])
nom_test = param['nom_test']                        # NAME OF THE CURRENT TEST
archi = param['archi']                              # NETWORK ARCHITECTURE
new_size = param['new_size']                        # SIZE DATA
bs = param['batch_size']                            # BATCH SIZE
zone = "NW"                                         # NW, SE
years = [int(args.nargs[2])]                        # 2016, 2017, 2018
months = [int(args.nargs[3])]                       # 1..12
parts_month = [1,2,3]                               # 1,2,3 (each month is divided in 3 parts)
input_timeframes = param['input_timeframes']                   # how many timeframes for input
output_timeframes = param['output_timeframes']                 # how many timeframes for output
overlapping_data = param['overlapping']                        # data overlap in time (= 1) or not (= 0)
rainfall_threshold_value = 40.                                 # Value above which values are considered to be one
size_regions = param["size_regions"]
threshold_rain_in_regions = param["threshold_rain_in_regions"]
#
features_bool = {'reflectivity': param['reflectivity_on'],
                 'rainfall quality': param['quality_on'],
                 'land sea': param['lsm_on'],
                 'elevation': param['relief_on']}
features_max_threshold = {'reflectivity': 60,
                          'rainfall quality': 100,
                          'land sea': 1,
                          'elevation': 600}
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
if archi=='ddnet':
    X_content = get_content_data(X)
N, T, H, W, C = X.shape


#########################################################################################
###################################### LOAD MODEL #######################################
#########################################################################################
model = tf.keras.models.load_model('models/'+nom_test+'.h5',
            custom_objects={'masked_ssim': masked_ssim, 'masked_psnr': masked_psnr, 'masked_acc': masked_acc, 'masked_BMW': masked_BMW,
                            'masked_cor': masked_cor, 'masked_recall': masked_recall, 'masked_prec': masked_prec,
                            'masked_mae': masked_mae, 'masked_mse': masked_mse, 'masked_logcosh': masked_logcosh})
if archi=='convdlrm':
    results = model.evaluate(X, y, batch_size=bs, return_dict=True)
elif archi=='ddnet':
    results = model.evaluate([X, X_content], y, batch_size=bs, return_dict=True)
with open('models/'+nom_test+'_new_test.csv', 'w') as f:
    w = csv.DictWriter(f, results.keys())
    w.writeheader()
    w.writerow(results)


### SHUFFLE
#perm = np.random.permutation(X.shape[0])
#X = tf.gather(X, perm, axis=0)
#y = tf.gather(y, perm, axis=0)
#if archi=='ddnet':
#    X_content = tf.gather(X_content, perm, axis=0)


#########################################################################################
####################################### PLOT TEST #######################################
#########################################################################################
foldername = FOLDERNAME+'plots/'
for itest in range(0, 100*nb_test, 100):
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
               [size_regions,size_regions], input_timeframes, output_timeframes,
               lat, lon, nom_test, archi, tag='test_'+str(itest),
               save=True, foldername=foldername)

