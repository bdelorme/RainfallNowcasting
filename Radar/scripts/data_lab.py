import numpy as np
import tensorflow as tf
from utils import *
import datetime
import xarray as xr
from os import path
import pandas as pd


#####################################################################################################################
#####################################################################################################################


def get_data_and_features(directory, zone, years, months, parts_month, new_size,
                          input_timeframes, output_timeframes,
                          rainfall_threshold_value, features_bool, weather_model_bool,
                          model, weather_model_max_threshold, weather_model_min_threshold,
                          features_max_threshold, features_min_threshold, overlapping_data,
                          threshold_rain_in_regions, size_regions):
  X, y, X_dates, X_regions = get_rainfall(rainfall_threshold_value,
                                directory, years, months, parts_month, zone, new_size,
                                input_timeframes, output_timeframes, overlapping_data,
                                threshold_rain_in_regions, size_regions)
  print('Got rainfall')
  features_list = []
  if features_bool['reflectivity'] == 1:
    reflectivity = get_reflectivity_quality("reflectivity",
                                            features_max_threshold['reflectivity'],
                                            directory, years, months, parts_month, zone,
                                            new_size, input_timeframes, output_timeframes,
                                            overlapping_data, X_dates, X_regions,
                                            size_regions)
    X = add_new_channel(X, reflectivity)
    features_list.append('reflectivity')
    print('Got reflectivity')
  if features_bool['rainfall quality'] == 1:
    quality = get_reflectivity_quality("quality",
                                        features_max_threshold['rainfall quality'],
                                        directory, years, months, parts_month, zone,
                                        new_size, input_timeframes, output_timeframes,
                                        overlapping_data, X_dates, X_regions,
                                        size_regions)
    X = add_new_channel(X, quality)
    features_list.append('quality')
    print('Got rainfall quality')
  if features_bool['land sea'] == 1:
    lsm = get_lsm_relief_mask(directory, zone, new_size, "LAND_GDS0_SFC",
                              features_max_threshold['land sea'],
                              features_min_threshold['land sea'],
                              X_regions, size_regions, input_timeframes)
    X = add_new_channel(X, lsm)
    features_list.append('land sea mask')
    print('Got land sea')
  if features_bool['elevation'] == 1:
    elevation = get_lsm_relief_mask(directory , zone, new_size, 'DIST_GDS0_SFC',
                                    features_max_threshold['elevation'],
                                    features_min_threshold['elevation'],
                                    X_regions, size_regions, input_timeframes)
    X = add_new_channel(X, elevation)
    features_list.append('elevation')
    print('Got elevation')
  if (model=='arpege') or (model=='arome'):
    model_data_dict = get_model_data(directory, zone, model, weather_model_bool, X_dates, new_size,
                                     weather_model_max_threshold, weather_model_min_threshold,
                                     X_regions, size_regions)
    for key in model_data_dict.keys():
      X = add_new_channel(X, model_data_dict[key])
      features_list.append(key)
    print('Got model data')
  return X, y, features_list


#####################################################################################################################
#####################################################################################################################


def get_rainfall(max_threshold_value, directory,
                 years, months, parts_month, zone, new_size,
                 input_timeframes, output_timeframes, overlapping_data,
                 threshold_rain_in_regions, size_regions):
  X = np.array([])
  first_part = True
  for year in years:
    for month in months:
      for part_month in parts_month:
        fname = directory + f'rainfall_{zone}/rainfall_{zone}_{str(year)}_{str(month).zfill(2)}.{str(part_month)}.npz'
        d = np.load(fname, allow_pickle=True)
        if first_part == True:
          data_temp = d['data']
          dates_temp = d['dates']
          first_part = False
        else:
          data_temp = np.concatenate((data_temp, d['data']), axis=0)
          dates_temp = np.concatenate((dates_temp, d['dates']), axis=0)

        num_of_timestep = data_temp.shape[0]
        num_of_sequences = num_of_timestep//(input_timeframes + output_timeframes)
        data_to_analyze = data_temp[:num_of_sequences*(input_timeframes+output_timeframes)]
        data_temp = data_temp[num_of_sequences*(input_timeframes+output_timeframes):]
        dates_to_analyze = dates_temp[:num_of_sequences*(input_timeframes+output_timeframes)]
        dates_temp = dates_temp[num_of_sequences*(input_timeframes+output_timeframes):]

        data_to_analyze = np.expand_dims(data_to_analyze, -1) # add channel dimension for tf handling (channel=1, ie grayscale)
        data_to_analyze[data_to_analyze == -1] = -10000
        # Resize the dimensions of the images to new_size
        if len(new_size) > 0:
          data_to_analyze = tf.image.resize(data_to_analyze, new_size)
          data_to_analyze = np.asarray(data_to_analyze)
        # Interpolate frame where values are missing
        if d['miss_dates'].shape[0] != 0:
          data_to_analyze, dates_to_analyze = interpolate_missing_data(data_to_analyze, d['miss_dates'], dates_to_analyze)

        data_to_analyze = data_to_analyze/max_threshold_value # normalize between 0 and 1 (min-max scaling)
        data_to_analyze[data_to_analyze > max_threshold_value] = 1
        data_to_analyze[data_to_analyze < 0] = -1
        X_to_analyze, y_to_analyze, X_dates_to_analyze, X_regions_to_analyze = multivariate_data(data_to_analyze, input_timeframes, output_timeframes, overlapping_data, dates_to_analyze, "rainfall", threshold_rain_in_regions, size_regions)
        # Store everything in one dataset
        if X.shape[0] == 0:
          X = X_to_analyze
          X_dates = X_dates_to_analyze
          y = y_to_analyze
          X_regions = X_regions_to_analyze
        else:
          X = np.vstack((X, X_to_analyze))
          X_dates = np.concatenate([X_dates, X_dates_to_analyze])
          y = np.vstack((y, y_to_analyze))
          X_regions = np.vstack((X_regions, X_regions_to_analyze))
        print("Year: {} Month: {} Part of the month: {}, Done !".format(year, month, part_month))
  X = tf.cast(X, tf.float32)
  y = tf.cast(y, tf.float32)
  return X, y, X_dates, X_regions

def get_reflectivity_quality(tag, max_threshold_value, directory,
                                      years, months, parts_month, zone, new_size,
                                      input_timeframes, output_timeframes, overlapping_data,
                                      rainfall_dates, rainfall_regions, size_regions):
  data = np.array([])
  for year in years:
    for month in months:
      for part_month in parts_month:
        if tag == 'reflectivity':
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
                                    overlapping_data, dates, tag=tag)
  X = keep_same_regions_as_rainfall_for_reflectivity_quality(X, X_dates, rainfall_dates, rainfall_regions, size_regions)
  X = tf.cast(X, tf.float32)
  return X

def multivariate_data(dataset, input_timeframes, output_timeframes, overlapping_data, dates, tag, threshold_rain_in_regions=100,
                      size_regions=64):
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

  if tag == "rainfall":
    X, y, X_dates, X_regions = keep_rainy_regions(X, X_dates, y, threshold_rain_in_regions, size_regions)
    return X, y, X_dates, X_regions
  return X, y, X_dates

def keep_rainy_regions(X, X_dates, y, threshold, size_regions):
  X = tf.cast(X, tf.float32)
  N, T, H, W, C = X.shape
  filters = np.ones([T, size_regions, size_regions, C, 1])
  total_rain = tf.nn.conv3d(X, filters, strides=(1,T,size_regions, size_regions,1), padding='VALID')
  regions_to_keep = tf.where(total_rain>threshold)
  X_numpy = X.numpy()
  X_new_regions = np.zeros([regions_to_keep.shape[0], T, size_regions, size_regions, C])
  y_new_regions = np.zeros([regions_to_keep.shape[0], y.shape[1], size_regions, size_regions, y.shape[4]])
  for i in range(regions_to_keep.shape[0]):
    X_new_regions[i] = X_numpy[regions_to_keep[i, 0], :, size_regions*regions_to_keep[i, 2]:size_regions*(regions_to_keep[i, 2]+1), size_regions*regions_to_keep[i, 3]:size_regions*(regions_to_keep[i, 3]+1), :]
    y_new_regions[i] = y[regions_to_keep[i, 0], :, size_regions*regions_to_keep[i, 2]:size_regions*(regions_to_keep[i, 2]+1), size_regions*regions_to_keep[i, 3]:size_regions*(regions_to_keep[i, 3]+1), :]
  X_new_regions_dates = X_dates[regions_to_keep[:,0]]
  X_new_regions_regions = regions_to_keep[:, 2:4]
  
  return X_new_regions, y_new_regions, X_new_regions_dates, X_new_regions_regions

def keep_same_regions_as_rainfall_for_reflectivity_quality(X, X_dates, rainfall_dates, rainfall_regions, size_regions):
  new_X = np.zeros([rainfall_dates.shape[0], X.shape[1], size_regions, size_regions, X.shape[4]])
  for i in range(rainfall_dates.shape[0]):
    idx = np.where(rainfall_dates[i,0] == X_dates[:,0])[0]
    new_X[i] = X[idx, :, size_regions*rainfall_regions[i,0]: size_regions*(rainfall_regions[i,0] + 1), size_regions*rainfall_regions[i,1]: size_regions*(rainfall_regions[i,1] + 1), :]
  return new_X

#####################################################################################################################
#####################################################################################################################


def get_content_data(X):
  return X[:, -1, :, :, :]


#####################################################################################################################
#####################################################################################################################


def get_lsm_relief_mask(directory, zone, new_size, mask_name, max_threshold, min_threshold, rainfall_regions, size_regions, input_timeframes):
  fname = directory + f'masks_{zone}.nc'
  data = xr.open_dataset(fname)
  lsm_mask = data[mask_name].values
  lsm_mask = (lsm_mask - min_threshold)/np.max(max_threshold)
  lsm_mask = np.expand_dims(lsm_mask, -1)
  if len(new_size) > 0:
          lsm_mask = tf.image.resize(lsm_mask, new_size)
  lsm_mask = convert_mask_to_rainfall_size(lsm_mask, rainfall_regions, size_regions, input_timeframes)
  lsm_mask = tf.cast(lsm_mask, tf.float32)
  return lsm_mask

def convert_mask_to_rainfall_size(lsm_mask, rainfall_regions, size_regions, input_timeframes):
  new_lsm_mask = np.zeros([rainfall_regions.shape[0], input_timeframes, size_regions, size_regions, lsm_mask.shape[2]])
  for i in range(rainfall_regions.shape[0]):
    new_lsm_mask[i] = np.tile(lsm_mask[size_regions*rainfall_regions[i,0]:size_regions*(rainfall_regions[i,0]+1), size_regions*rainfall_regions[i,1]:size_regions*(rainfall_regions[i,1]+1), :], (input_timeframes, 1, 1, 1))
  return new_lsm_mask


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
                   weather_model_max_threshold, weather_model_min_threshold, rainfall_regions,
                   size_regions):
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
        if level !=  'PRECIP':
          if data.step.values.shape[0] != 25:
            data = interpolate_in_time(data, 25)
        else:
          if data.step.values.shape[0] !=24:
            data= interpolate_in_time(data, 24)
      else:
        data = create_missing_array(directory, zone, model, level, date)
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
    print(rainfall_regions.shape)
    print(model_data_dict[i].shape)
    new_model_data = np.zeros([model_data_dict[i].shape[0], model_data_dict[i].shape[1], size_regions, size_regions, model_data_dict[i].shape[4]])
    for j in range(rainfall_regions.shape[0]):
      new_model_data[j] = model_data_dict[i][j, :, size_regions*rainfall_regions[j,0]:size_regions*(rainfall_regions[j,0]+1), size_regions*rainfall_regions[j,1]:size_regions*(rainfall_regions[j,1] + 1), :]
    model_data_dict[i] = new_model_data
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
  X_dates_rounded_to_day = np.vectorize(round_to_day)(X_dates)
  unique_dates_rounded_to_day, day_to_X = np.unique(X_dates_rounded_to_day, return_inverse=True)
  day_to_X = day_to_X.reshape(X_dates.shape)
  X_dates_hour = np.vectorize(get_hour)(X_dates)
  return unique_dates_rounded_to_day, day_to_X, X_dates_hour

def interpolate_in_time(array, num_timesteps):
  H = array.dims['latitude']
  W = array.dims['longitude']
  array_times = array['valid_time'].values
  date = array['time'].values
  array_times_pd = pd.to_datetime(array_times)
  date_pd = pd.to_datetime(date)
  hours = list(array_times_pd.hour)
  if array_times_pd[-1].day > date_pd.day:
    hours[-1] = num_timesteps - 1
  final_var = {}
  for var in array.data_vars:
    new_array = np.zeros([num_timesteps, H, W])
    for h in range(len(hours)):
      new_array[hours[h]] = array[var][h]
    for h in range(num_timesteps):
      if h not in hours:
        if h < hours[0]:
          new_array[h] = array[var][0]
        elif h > hours[-1]:
          new_array[h] = array[var][len(hours) - 1]
        else:
          m = np.argmin(abs(np.array(hours) - h))
          if hours[m] < h:
            new_array[h] = (hours[m+1] - h)/(hours[m + 1]- hours[m])*array[var][m] + (h-hours[m])/(hours[m+1]-hours[m])*array[var][m+1]
          else:
            new_array[h] = (hours[m] - h)/(hours[m]- hours[m-1])*array[var][m-1] + (h-hours[m-1])/(hours[m]-hours[m-1])*array[var][m]
    final_var[var] = (("step", "latitude", "longitude"), new_array)
  new_dates = np.array(pd.date_range(date_pd, periods=num_timesteps, freq='H'))
  new_xarray = xr.Dataset(
     final_var, coords={"latitude": array["latitude"], "longitude": array["longitude"],
             "step": np.array(new_dates-new_dates[0], dtype='timedelta64[ns]'),
             "valid_time": new_dates, "time": array["time"]})
  return new_xarray


def create_missing_array(directory, zone, model, level, date):
  if level == '2m':
    params = ['t2m', 'd2m', 'r']
  elif level == '10m':
    params = ['ws', 'p3031', 'u10', 'v10']
  elif level == 'P_sea_level':
    params = ['msl']
  else:
    params = ['tp']
  date_prev = date + datetime.timedelta(days = -1)
  fname_prev = directory + f'{model}/{level}/{model}_{level}_{zone}_{date_prev.year}{str(date_prev.month).zfill(2)}{str(date_prev.day).zfill(2)}000000.grib'
  days_prev = 1
  while path.exists(fname_prev) == False:
    date_prev = date_prev + datetime.timedelta(days = -1)
    fname_prev = directory + f'{model}/{level}/{model}_{level}_{zone}_{date_prev.year}{str(date_prev.month).zfill(2)}{str(date_prev.day).zfill(2)}000000.grib'
    days_prev += 1
  data_prev = xr.open_dataset(fname_prev, engine='cfgrib')
  days_next = 1
  date_next = date + datetime.timedelta(days = 1)
  fname_next = directory + f'{model}/{level}/{model}_{level}_{zone}_{date_next.year}{str(date_next.month).zfill(2)}{str(date_next.day).zfill(2)}000000.grib'
  while path.exists(fname_next) == False:
    date_next = date_next + datetime.timedelta(days = 1)
    fname_next = directory + f'{model}/{level}/{model}_{level}_{zone}_{date_next.year}{str(date_next.month).zfill(2)}{str(date_next.day).zfill(2)}000000.grib'
    days_next +=1
  data_next = xr.open_dataset(fname_next, engine='cfgrib')
  if level !=  'PRECIP':
    if data_prev.step.values.shape[0] != 25:
      data_prev = interpolate_in_time(data_prev, 25)
  else:
    if data_prev.step.values.shape[0] !=24:
      data_prev = interpolate_in_time(data_prev, 24)
  if level !=  'PRECIP':
    if data_next.step.values.shape[0] != 25:
      data_next = interpolate_in_time(data_next, 25)
  else:
    if data_next.step.values.shape[0] !=24:
      data_next = interpolate_in_time(data_next, 24)
  data = data_prev.copy()
  for param in params:
    data[param] = days_next/(days_prev + days_next)*data_prev[param] + days_prev/(days_prev + days_next)*data_next[param]
  data = data.assign_coords({'time' : date})
  data = data.assign_coords({'valid_time': data['valid_time'].values + np.timedelta64(1,'D')})
  return data


#####################################################################################################################
#####################################################################################################################
