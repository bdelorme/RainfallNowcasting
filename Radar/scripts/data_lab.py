import numpy as np
import tensorflow as tf
from utils import *
import datetime
import xarray as xr
from os import path


#####################################################################################################################
#####################################################################################################################


def get_data_and_features(directory, zone, years, months, parts_month, new_size,
                          input_timeframes, output_timeframes, percentage_test, 
                          rainfall_threshold_value, features_bool, weather_model_bool, 
                          model, weather_model_max_threshold, weather_model_min_threshold, 
                          features_max_threshold, features_min_threshold, overlapping_data):
  X, y, X_dates = get_rainfall_reflectivity_quality("rainfall", rainfall_threshold_value, 
                                                    directory, years, months, parts_month, zone, new_size, 
                                                    input_timeframes, output_timeframes, overlapping_data)
  print('Got rainfall')
  features_list = []
  if features_bool['reflectivity'] == 1:
    reflectivity = get_rainfall_reflectivity_quality("reflectivity", 
                                                     features_max_threshold['reflectivity'], 
                                                     directory, years, months, parts_month, zone, 
                                                     new_size, input_timeframes, output_timeframes, 
                                                     overlapping_data)
    X = add_new_channel(X, reflectivity)
    features_list.append('reflectivity')
    print('Got reflectivity')
  if features_bool['rainfall quality'] == 1:
    quality = get_rainfall_reflectivity_quality("quality",
                                                features_max_threshold['rainfall quality'], 
                                                directory, years, months, parts_month, zone, 
                                                new_size, input_timeframes, output_timeframes, 
                                                overlapping_data)
    X = add_new_channel(X, quality)
    features_list.append('quality')
    print('Got rainfall quality')
  if features_bool['land sea'] == 1:
    lsm = get_lsm_relief_mask(directory, zone, new_size, "LAND_GDS0_SFC", 
                              features_max_threshold['land sea'], 
                              features_min_threshold['land sea'])
    X = add_new_channel(X, lsm)
    features_list.append('land sea mask')
    print('Got land sea')
  if features_bool['elevation'] == 1:
    elevation = get_lsm_relief_mask(directory , zone, new_size, 'DIST_GDS0_SFC', 
                                    features_max_threshold['elevation'], 
                                    features_min_threshold['elevation'])
    X = add_new_channel(X, elevation)
    features_list.append('elevation')
    print('Got elevation')
  if (model=='arpege') or (model=='arome'):
    model_data_dict = get_model_data(directory, zone, model, weather_model_bool, X_dates, new_size, 
                                     weather_model_max_threshold, weather_model_min_threshold)
    for key in model_data_dict.keys():
      X = add_new_channel(X, model_data_dict[key])
      features_list.append(i)
    print('Got model data')
  return X, y, features_list


#####################################################################################################################
#####################################################################################################################


def get_rainfall_reflectivity_quality(tag, max_threshold_value, directory, 
                                      years, months, parts_month, zone, new_size, 
                                      input_timeframes, output_timeframes, overlapping_data):
  data = np.array([])
  for year in years:
    for month in months:
      for part_month in parts_month:
        if tag == 'rainfall':
          fname = directory + f'{tag}_{zone}/{tag}_{zone}_{str(year)}_{str(month).zfill(2)}.{str(part_month)}.npz'
          d = np.load(fname, allow_pickle=True)
          data_temp = d['data']
          dates_temp = d['dates']
          data_temp = np.expand_dims(data_temp, -1) # add channel dimension for tf handling (channel=1, ie grayscale)
          data_temp[data_temp == -1] = -10000
        elif tag == 'reflectivity':
          if year == 2018 and month > 2:
            new_or_old = "new"
          else:
            new_or_old = "old"
          fname = directory + f'{tag}_{zone}/{tag}_{new_or_old}_{zone}_{str(year)}_{str(month).zfill(2)}.{str(part_month)}.npz'
          d = np.load(fname, allow_pickle=True)
          data_temp = d['data']
          dates_temp = d['dates']
          data_temp = np.expand_dims(data_temp, -1) # add channel dimension for tf handling (channel=1, ie grayscale)
          data_temp = data_temp.copy()
          data_temp[data_temp == 255] = -10000
        elif tag == 'quality':
          fname_mean = directory + f'rainfall_{tag}_{zone}/mean/rainfall_mean_{tag}-code_{zone}_{str(year)}_{str(month).zfill(2)}.{str(part_month)}.npz'
          fname_diff = directory + f'rainfall_{tag}_{zone}/diff/rainfall_diff_{tag}-code_{zone}_{str(year)}_{str(month).zfill(2)}.{str(part_month)}.npz'
          d_mean = np.load(fname_mean, allow_pickle=True)
          d = np.load(fname_diff, allow_pickle=True)
          data_temp = d['data']
          dates_temp = d['dates']
          day = np.vectorize(get_day)(d['dates'])
          day_mean = np.vectorize(get_day)(d_mean['dates'])
          data_temp += d_mean['data'][np.where(day[:, None] == day_mean[None, :])[1]]
          data_temp = np.expand_dims(data_temp, -1) # add channel dimension for tf handling (channel=1, ie grayscale)
          data_temp = data_temp.copy()
          data_temp[data_temp == 255] = -10000
        # Resize the dimensions of the images to new_size
        if len(new_size) > 0:
          data_temp = tf.image.resize(data_temp, new_size)
          data_temp = np.asarray(data_temp)
        # Interpolate frame where values are missing
        if d['miss_dates'].shape[0] != 0:
          data_temp, dates_temp = interpolate_missing_data(data_temp, d['miss_dates'], dates_temp)
        # Store everything in one dataset
        if data.shape[0] == 0:
          data = data_temp
          dates = dates_temp
        else:
          data = np.vstack((data, data_temp))
          dates = np.concatenate([dates, dates_temp])
        print("Year: {} Month: {} Part of the month: {}, Done !".format(year, month, part_month))
  # Normalize between 0 and 1 (min-max scaling)
  data = data/max_threshold_value
  data[data > max_threshold_value] = 1
  data[data < 0] = -1
  # Get I/O
  X, y, X_dates = multivariate_data(data, input_timeframes, output_timeframes, 
                                    overlapping_data, dates)
  if tag == 'rainfall':
    X = tf.cast(X, tf.float32)
    y = tf.cast(y, tf.float32)
    return X, y, X_dates
  else:
    X = tf.cast(X, tf.float32)
    return X

def multivariate_data(dataset, input_timeframes, output_timeframes, overlapping_data, dates):
  X = []
  y = []
  X_dates = []
  if overlapping_data == 0:
    step = input_timeframes + output_timeframes
  else:
    step = 1
  for i in range(0, dataset.shape[0]-(input_timeframes + output_timeframes), step):
    indices_X = range(i, i + input_timeframes)
    indices_y = range(i + input_timeframes, i + input_timeframes + output_timeframes)
    X.append(dataset[indices_X])
    X_dates.append(dates[indices_X])
    y.append(dataset[indices_y])
  X = np.array(X)
  X_dates = np.array(X_dates)
  y = np.array(y)
  return X, y, X_dates


#####################################################################################################################
#####################################################################################################################


def get_content_data(X):
  return X[:, -1, :, :, :]


#####################################################################################################################
#####################################################################################################################


def get_lsm_relief_mask(directory, zone, new_size, mask_name, max_threshold, min_threshold):
  fname = directory + f'masks_{zone}.nc'
  data = xr.open_dataset(fname)
  lsm_mask = data[mask_name].values
  lsm_mask = (lsm_mask - min_threshold)/np.max(max_threshold)
  lsm_mask = np.expand_dims(lsm_mask, -1)
  if len(new_size) > 0:
          lsm_mask = tf.image.resize(lsm_mask, new_size)
  lsm_mask = tf.cast(lsm_mask, tf.float32)
  return lsm_mask


#####################################################################################################################
#####################################################################################################################


def split_train_test(percentage_test, X, y):
  nX, ny = unison_shuffle(X, y)
  N = ny.shape[0]
  size_train = int(np.ceil((1-percentage_test)*N))
  y_train = ny[:size_train]
  y_test = ny[size_train:]
  X_train = nX[:size_train]
  X_test = nX[size_train:]
  return X_train, y_train, X_test, y_test


#####################################################################################################################
#####################################################################################################################


def get_model_data(directory_dataset, zone, model, weather_model_bool, X_dates, size, 
                   weather_model_max_threshold, weather_model_min_threshold):
  N, T = X_dates.shape
  H, W = size
  model_data_dict = {}
  levels, params, params_to_name = get_levels_params(weather_model_bool)
  unique_dates_rounded_to_day, day_to_X, X_dates_hour = get_dates_of_interest(X_dates)
  for i in range(len(levels)):
    level = levels[i]
    for d in range(unique_dates_rounded_to_day.shape[0]):
      date = unique_dates_rounded_to_day[d]
      directory = directory_dataset + 'models_2D_' + zone + '/'
      fname = directory + f'{model}/{level}/{model}_{level}_{zone}_{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}000000.grib'
      if path.exists(fname):
        data = xr.open_dataset(fname, engine='cfgrib') 
      else:
        fname = directory + f'{model}/{level}/{model}_{level}_{zone}_{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}000000.nc'
        data = xr.open_dataset(fname, engine='netcdf4')
      for param in params[i]:
        if params_to_name[param] not in model_data_dict.keys():
          model_data_dict[params_to_name[param]] = -np.ones([N, T, H, W, 1])
        feat = data[param].values
        feat = np.moveaxis(feat, 0, 2)
        feat = tf.image.resize(feat, [H,W])
        feat = np.asarray(feat)
        feat = np.moveaxis(feat, 2, 0)
        feat = np.expand_dims(feat, -1)
        pos_X_with_date = np.where(day_to_X == d)
        model_data_dict[params_to_name[param]][pos_X_with_date] = feat[X_dates_hour[pos_X_with_date]]
      print(date)
    print(level)
  if weather_model_bool['wind components'] == 1:
    model_data_dict['wind components'] = np.concatenate([model_data_dict['wind components 1'], model_data_dict['wind components 2']], axis = -1)
    del model_data_dict['wind components 1']
    del model_data_dict['wind components 2']
  for i in model_data_dict.keys():
    model_data_dict[i] -= weather_model_min_threshold[i]
    model_data_dict[i] /= weather_model_max_threshold[i]
    model_data_dict[i] = tf.cast(model_data_dict[i], tf.float32)
  return model_data_dict

def get_levels_params(weather_model_bool):
  levels = []
  params = []
  params_to_name = {}
  if weather_model_bool['temperature'] == 1 or weather_model_bool['dew point temperature'] == 1 or weather_model_bool['humidity'] == 1:
    levels.append('2m')
  if weather_model_bool['wind speed'] or weather_model_bool['wind directions'] == 1 or weather_model_bool['wind components'] == 1:
    levels.append('10m')
  if weather_model_bool['pressure'] == 1:
    levels.append('P_sea_level')
  if weather_model_bool['precipitation'] == 1:
    levels.append('PRECIP')
  for i in range(len(levels)):
    params_i = []
    if levels[i] == '2m':
      if weather_model_bool['temperature'] == 1:
        params_i.append('t2m')
        params_to_name['t2m'] = 'temperature'
      if weather_model_bool['dew point temperature'] == 1:
        params_i.append('d2m')
        params_to_name['d2m'] = 'dew point temperature'
      if weather_model_bool['humidity'] == 1:
        params_i.append('r')
        params_to_name['r'] = 'humidity'
    if levels[i] == '10m':
      if weather_model_bool['wind speed'] == 1:
        params_i.append('ws')
        params_to_name['ws'] = 'wind speed'
      if weather_model_bool['wind directions'] == 1:
        params_i.append('p3031')
        params_to_name['p3031'] = 'wind directions'
      if weather_model_bool['wind components'] == 1:
        params_i.append('u10')
        params_i.append('v10')
        params_to_name['u10'] = 'wind components 1'
        params_to_name['v10'] = 'wind components 2'
    if levels[i] == 'P_sea_level':
      params_i.append('msl')
      params_to_name['msl'] = 'pressure'
    if levels[i] == 'PRECIP':
      params_i.append('tp')
      params_to_name['tp'] = 'precipitation'
    params.append(params_i)
  return levels, params, params_to_name

def get_dates_of_interest(X_dates):
  year_month_day_hour = np.array([])
  X_dates_rounded_to_day = np.vectorize(round_to_day)(X_dates)
  unique_dates_rounded_to_day, day_to_X = np.unique(X_dates_rounded_to_day, return_inverse=True)
  day_to_X = day_to_X.reshape(X_dates.shape)
  X_dates_hour = np.vectorize(get_hour)(X_dates)
  return unique_dates_rounded_to_day, day_to_X, X_dates_hour
      

#####################################################################################################################
#####################################################################################################################