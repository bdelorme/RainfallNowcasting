import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
import xarray as xr
import os.path
from os import path
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

def get_coords(directory, zone):
  fname_coords = directory + f'radar_coords_{zone}.npz'
  #get the coordinates of the points
  coords = np.load(fname_coords, allow_pickle=True)
  #it is about coordinates of the top left corner of pixels -> it is necessary to get the coordinates of the center of pixels
  #to perform a correct overlay of data
  resolution = 0.01 #spatial resolution of radar data (into degrees)
  lat = coords['lats']-resolution/2
  lon = coords['lons']+resolution/2
  return [lat, lon]

def get_data_and_features(directory, zone, years, months, part_months, new_size,
                          input_timeframes, output_timeframes, overlapping_data,
                          normalization_min, percentage_test,
                          rainfall_threshold_value,
                          features_bool, weather_model_bool, model,
                          weather_model_max_threshold, weather_model_min_threshold,
                          features_max_threshold, features_min_threshold):
  X, y, y_mask, X_dates = get_rainfall_reflectivity_quality("rainfall", rainfall_threshold_value, directory, years, months, part_months, zone, new_size, input_timeframes, output_timeframes, overlapping_data, normalization_min)
  N, T, H, W, C = X.shape
  print('Got rainfall')
  features_dict = {}
  if features_bool['reflectivity'] == 1:
    features_dict['reflectivity'] = get_rainfall_reflectivity_quality("reflectivity", features_max_threshold['reflectivity'], directory, years, months, part_months, zone, new_size, input_timeframes, output_timeframes, overlapping_data, normalization_min)
    print('Got reflectivity')
  if features_bool['rainfall quality'] == 1:
    features_dict['rainfall quality'] = get_rainfall_reflectivity_quality("quality", 100, directory, years, months, part_months, zone, new_size, input_timeframes, output_timeframes, overlapping_data, normalization_min)
    print('Got rainfall quality')
  if features_bool['land sea'] == 1:
    features_dict['land sea'] = get_lsm_relief_mask(directory, zone, [H,W], "LAND_GDS0_SFC", normalization_min, features_max_threshold['land sea'], features_min_threshold['land sea'])
    print('Got land sea')
  if features_bool['elevation'] == 1:
    features_dict['elevation'] = get_lsm_relief_mask(directory , zone, [H,W], 'DIST_GDS0_SFC', normalization_min, features_max_threshold['elevation'], features_min_threshold['elevation'])
    print('Got elevation')
  model_data_dict = get_model_data(directory, zone, model, weather_model_bool, X_dates, [H,W], normalization_min, weather_model_max_threshold, weather_model_min_threshold)
  print('Got model data')
  for i in model_data_dict.keys():
    features_dict[i] = model_data_dict[i]
  
  return X, y, features_dict

def get_rainfall_reflectivity_quality(tag, max_threshold_value, directory, years, months, part_months, zone, new_size, input_timeframes, output_timeframes, overlapping_data, normalization_min):
  data = np.array([])
  for year in years:
    for month in months:
      for part_month in part_months:
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

        #Store everything in one dataset
        if data.shape[0] == 0:
          data = data_temp
          dates = dates_temp
        else:
          data = np.vstack((data, data_temp))
          dates = np.concatenate([dates, dates_temp])
        
        print("Year: {} Month: {} Part of the month: {}, Done !".format(year, month, part_month))

  data = data/max_threshold_value # normalize between 0 and 1 (min-max scaling)
  data[data > max_threshold_value] = 1
  data[data < 0] = -1
  if normalization_min == -1:
    data = 2*(data - 0.5)
  X, y, y_mask, X_dates = multivariate_data(data, 0, None, input_timeframes, output_timeframes, overlapping_data, normalization_min, dates)
  if tag == 'rainfall':
    X = tf.cast(X, tf.float32)
    y = tf.cast(y, tf.float32)
    y_mask = tf.cast(y_mask, tf.float32)
    return X, y, y_mask, X_dates
  else:
    X = tf.cast(X, tf.float32)
    return X

def interpolate_missing_data(data, miss_dates, dates):
  for i in range(miss_dates.shape[0]):
    if miss_dates[i] < dates[0]:
      data = np.insert(data, 0, data[0], axis=0)
      dates = np.insert(dates, 0, miss_dates[i])
    elif miss_dates[i] > dates[-1]:
      data = np.insert(data, -1, data[0], axis=0)
      dates = np.inster(dates, -1, miss_dates[i])
    else:
      m = np.argmin(abs(dates - miss_dates[i]))
      if dates[m] < miss_dates[i]:
        data = np.insert(data, m+1, (data[m] + data[m+1])/2, axis=0)
        dates = np.insert(dates, m+1, miss_dates[i])
      else:
        data = np.insert(data, m, (data[m] + data[m-1])/2, axis=0)
        dates = np.insert(dates, m, miss_dates[i])
  return data, dates

# A function to help get the training and validation data correctly
def multivariate_data(dataset, start_index, end_index, input_timeframes, output_timeframes, overlapping_data, normalization_min, dates):
  X = []
  y = []
  X_dates = []

  if end_index is None:
    end_index = dataset.shape[0]

  if overlapping_data == 0:
    step = input_timeframes + output_timeframes
  else:
    step = 1

  for i in range(start_index, end_index-(input_timeframes + output_timeframes), step):
    indices_X = range(i, i + input_timeframes)
    indices_y = range(i + input_timeframes, i + input_timeframes + output_timeframes)
    X.append(dataset[indices_X])
    X_dates.append(dates[indices_X])
    y.append(dataset[indices_y])
  
  X = np.array(X)
  X_dates = np.array(X_dates)
  y = np.array(y)
  y_mask = y.copy()
  y_mask[y_mask >= normalization_min] = 1
  y_mask[y_mask < normalization_min] = 0
  return X, y, y_mask, X_dates

def convert_to_motion_content_data(X, X_dates):
  X_content = X[:, -1, :, :, :]
  X_motion = X[:, 1:, :, :, :] - X[:, :-1, :, :, :]
  X_content_dates = X_dates[:, -1]
  X_motion_dates = X_dates[:, 1:]
  return X_motion, X_content, X_motion_dates, X_content_dates

def get_lsm_relief_mask(directory, zone, new_size, mask_name, normalization_min, max_threshold, min_threshold):
  fname = directory + f'masks_{zone}.nc'
  data = xr.open_dataset(fname)
  lsm_mask = data[mask_name].values
  lsm_mask = (lsm_mask - min_threshold)/np.max(max_threshold)
  if normalization_min == -1:
    lsm_mask = 2*(lsm_mask - 0.5)
  lsm_mask = np.expand_dims(lsm_mask, -1)
  if len(new_size) > 0:
          lsm_mask = tf.image.resize(lsm_mask, new_size)
  lsm_mask = tf.cast(lsm_mask, tf.float32)
  return lsm_mask

def add_new_channel(X, new_channel):
  X_num_dim = len(X.shape)
  new_channel_num_dim = len(new_channel.shape)
  new_channel_temp = new_channel
  for i in range(X_num_dim - new_channel_num_dim - 1, -1, -1):
    new_channel_temp = tf.expand_dims(new_channel_temp, axis=0)
    new_channel_temp = tf.repeat(new_channel_temp, X.shape[i], axis=0)
  new_X = tf.concat([X, new_channel_temp], axis=-1)
  return new_X

def split_train_test(percentage_test, motion_content_data, X, y, motion_difference):
  N, _, _, _, _ = y.shape
  size_train = int(np.ceil((1-percentage_test)*N))
  y_train = y[:size_train]
  y_test = y[size_train:]
  X_train = X[:size_train]
  X_test = X[size_train:]
  if motion_content_data == 1:
    X_content_train = X_train[:, -1]
    X_content_test = X_train[:, -1]
    if motion_difference == 1:
      X_train = np.asarray(X_train).copy()
      X_motion = X_train[:, 1:, :, :, :]
      X_motion[:, :, :, :, 0] = X_train[:, 1:, :, :, 0] - X_train[:, :-1, :, :, 0]
      X_test = np.asarray(X_test).copy()
      X_motion_test = np.asarray(X_test[:, 1:])
      X_motion_test[:, :, :, :, 0] = X_test[:, 1:, :, :, 0] - X_test[:, :-1, :, :, 0]
      X_motion = tf.cast(X_motion, tf.float32)
      X_motion_test = tf.cast(X_motion_test, tf.float32)
    else:
      X_motion = X_train
      X_motion_test = X_test
    return X_motion, X_content_train, y_train, X_motion_test, X_content_test, y_test
  return X_train, y_train, X_test, y_test

def get_model_data(directory_dataset, zone, model, weather_model_bool, X_dates, size, normalization_min, weather_model_max_threshold, weather_model_min_threshold):
  N, T = X_dates.shape
  H = size[0]
  W = size[1]
  model_data_dict = {}
  levels = []
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
        #feat = np.array(data[param]).reshape((num_latitude, num_longitude, num_time))
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
    if normalization_min == -1:
      model_data_dict[i] = (model_data_dict[i] - 0.5)*2
    model_data_dict[i] = tf.cast(model_data_dict[i], tf.float32)
  return model_data_dict
      
def get_dates_of_interest(X_dates):
  year_month_day_hour = np.array([])
  X_dates_rounded_to_day = np.vectorize(round_to_day)(X_dates)
  unique_dates_rounded_to_day, day_to_X = np.unique(X_dates_rounded_to_day, return_inverse=True)
  day_to_X = day_to_X.reshape(X_dates.shape)
  X_dates_hour = np.vectorize(get_hour)(X_dates)
  return unique_dates_rounded_to_day, day_to_X, X_dates_hour

def round_to_day(dt):
    return dt + datetime.timedelta(hours = -dt.hour, minutes = -dt.minute, seconds = -dt.second)

def get_hour(dt):
  return dt.hour

def get_day(dt):
  return dt.day

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

def add_model_channel(X, model_data_dict, weather_model_position, motion_content_data, X_content=[]):
  for i in model_data_dict.keys():
    if weather_model_position[i] == 1:
      X = add_new_channel(X, model_data_dict[i])
      if motion_content_data == 1:
        X_content = add_new_channel(X_content, model_data_dict[i][:, -1])
  if motion_content_data == 1:
    return X, X_content
  return X

def add_features_as_channels(X, features_dict, weather_model_position, features_position):
  X_feat =X
  for i in features_dict.keys():
    if i in weather_model_position.keys() and weather_model_position[i] == 1:
      X_feat = add_new_channel(X_feat, features_dict[i])
    elif i in features_position.keys() and features_position[i] == 1:
      X_feat = add_new_channel(X_feat, features_dict[i])
  return X_feat

def interpolate_in_time(array, num_timesteps):
  T = array.dims['step']
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
      
      


