import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
import xarray as xr

def get_coords(directory, zone):
  fname_coords = directory + 'Radar_coords/Radar_coords/'+ f'radar_coords_{zone}.npz'
  #get the coordinates of the points
  coords = np.load(fname_coords, allow_pickle=True)
  #it is about coordinates of the top left corner of pixels -> it is necessary to get the coordinates of the center of pixels
  #to perform a correct overlay of data
  resolution = 0.01 #spatial resolution of radar data (into degrees)
  lat = coords['lats']-resolution/2
  lon = coords['lons']+resolution/2
  return [lat, lon]

def data_preprocess(directory, years, months, part_months, zone, new_size, ind_min, ind_max, input_timeframes, output_timeframes, overlapping_data, motion_content_data, normalization_min, rain_or_reflectivity, threshold_value):
  data = np.array([])
  for year in years:
    for month in months:
      for part_month in part_months:
        if rain_or_reflectivity == 0:
          fname = directory + f'{zone}_rainfall_{str(year)}/{zone}_rainfall_{str(year)}/rainfall-{zone}-{str(year)}-{str(month).zfill(2)}/rainfall-{zone}-{str(year)}-{str(month).zfill(2)}/rainfall_{zone}_{str(year)}_{str(month).zfill(2)}.{str(part_month)}.npz'
        else:
          fname = directory + f'{zone}_reflectivity_old_product_{str(year)}/{zone}_reflectivity_old_product_{str(year)}/reflectivity-old-{zone}-{str(year)}-{str(month).zfill(2)}/reflectivity-old-{zone}-{str(year)}-{str(month).zfill(2)}/reflectivity_old_{zone}_{str(year)}_{str(month).zfill(2)}.{str(part_month)}.npz'
        d = np.load(fname, allow_pickle=True)

        # Check if ind_min and ind_max are defined
        if ind_min != None and ind_max != None:
          data_temp = d['data'][ind_min:ind_max, :, :]
          dates_temp = d['dates'][ind_min:ind_max, :, :]
        else:
          data_temp = d['data']
          dates_temp = d['dates']

        data_temp = np.expand_dims(data_temp, -1) # add channel dimension for tf handling (channel=1, ie grayscale)

        if rain_or_reflectivity == 1:
          data_temp = data_temp.copy()
          data_temp[data_temp == 255] = -10000
        else:
          data_temp[data_temp == -1] = -10000
        
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

  data = data/threshold_value # normalize between 0 and 1 (min-max scaling)
  data[data > 1] = 1
  data[data < 0] = -1
  if normalization_min == -1:
    data = 2*(data - 0.5)
  X, y, y_mask, X_dates = multivariate_data(data, 0, None, input_timeframes, output_timeframes, overlapping_data, normalization_min, dates)
  if motion_content_data == 0:
    X = tf.cast(X, tf.float32)
    y = tf.cast(y, tf.float32)
    y_mask = tf.cast(y_mask, tf.float32)
    return X, y, y_mask, X_dates
  else:
    X_motion, X_content, X_motion_dates, X_content_dates = convert_to_motion_content_data(X, X_dates)
    X_motion = tf.cast(X_motion, tf.float32)
    X_content = tf.cast(X_content, tf.float32)
    y = tf.cast(y, tf.float32)
    y_mask = tf.cast(y_mask, tf.float32)
    return X_motion, X_content, y, y_mask, X_motion_dates, X_content_dates

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

def get_lsm_relief_mask(directory, zone, new_size, mask_name):
  fname = directory + f'{zone}_masks.pkl'
  data = pd.read_pickle(fname)
  num_latitude = np.unique(data.index.get_level_values(0).values).shape[0]
  num_longitude = np.unique(data.index.get_level_values(1).values).shape[0]
  lsm_mask = np.array(data[mask_name]).reshape((num_latitude, num_longitude))
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

def split_train_test(percentage_test, motion_content_data, X, y, X_content=[]):
  N, _, _, _, _ = y.shape
  size_train = int(np.ceil((1-percentage_test)*N))
  y_train = y[:size_train]
  y_test = y[size_train:]
  X_train = X[:size_train]
  X_test = X[size_train:]
  if motion_content_data == 1:
    X_content_train = X_content[:size_train]
    X_content_test = X_content[size_train:]
    return X_train, X_content_train, y_train, X_test, X_content_test, y_test
  return X_train, y_train, X_test, y_test

def get_model_data(directory_dataset, zone, model, weather_model_bool, X_dates, size):
  N, T = X_dates.shape
  H = size[0]
  W = size[1]
  model_data_dict = {}
  model_data_dict['model'] = model
  levels = []
  if model == 'arpege':
    MODEL = 'ARPEGE'
  else:
    MODEL = 'AROME'
  levels, params, params_to_name = get_levels_params(weather_model_bool)
  unique_dates_rounded_to_day, day_to_X, X_dates_hour = get_dates_of_interest(X_dates)
  for i in range(len(levels)):
    level = levels[i]
    for d in range(unique_dates_rounded_to_day.shape[0]):
      date = unique_dates_rounded_to_day[d]
      directory = directory_dataset + zone + '_weather_models_2D_parameters_' + str(date.year) + str(date.month).zfill(2) + '/' + str(date.year) + str(date.month).zfill(2) + '/'
      fname = directory + f'{MODEL}/{level}/{model}_{level}_{zone}_{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}000000.pkl'
      # data = xr.open_dataset(fname) 
      data = pd.read_pickle(fname)
      num_latitude = np.unique(data.index.get_level_values(0).values).shape[0]
      num_longitude = np.unique(data.index.get_level_values(1).values).shape[0]
      num_time = np.unique(data.index.get_level_values(2).values).shape[0]
      for param in params[i]:
        if params_to_name[param] not in model_data_dict.keys():
          model_data_dict[params_to_name[param]] = -np.ones([N, T, H, W, 1])
        feat = np.array(data[param]).reshape((num_latitude, num_longitude, num_time))
        feat = tf.image.resize(feat, [H,W])
        feat = np.asarray(feat)
        feat = np.moveaxis(feat, 2, 0)
        feat = np.expand_dims(feat, -1)
        pos_X_with_date = np.where(day_to_X == d)
        model_data_dict[params_to_name[param]][pos_X_with_date] = feat[X_dates_hour[pos_X_with_date]]
  if weather_model_bool['wind components'] == 1:
    model_data_dict['wind components'] = np.concatenate([model_data_dict['wind components 1'], model_data_dict['wind components 2']], axis = -1)
    del model_data_dict['wind components 1']
    del model_data_dict['wind components 2']
  for i in model_data_dict.keys():
    if i != 'model':
      model_data_dict[i] = tf.cast(model_data_dict[i], tf.float32)
  return model_data_dict
      
def get_dates_of_interest(X_dates):
  year_month_day_hour = np.array([])
  X_dates_rounded_to_day = np.vectorize(round_to_hour)(X_dates)
  unique_dates_rounded_to_day, day_to_X = np.unique(X_dates_rounded_to_day, return_inverse=True)
  day_to_X = day_to_X.reshape(X_dates.shape)
  X_dates_hour = np.vectorize(get_hour)(X_dates)
  return unique_dates_rounded_to_day, day_to_X, X_dates_hour

def round_to_hour(dt):
    return dt + datetime.timedelta(hours = -dt.hour, minutes = -dt.minute, seconds = -dt.second)

def get_hour(dt):
  return dt.hour

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
    if i != 'model':
      if weather_model_position[i] == 1:
        X = add_new_channel(X, model_data_dict[i])
        if motion_content_data == 1:
          X_content = add_new_channel(X_content, model_data_dict[i][:, -1])
  if motion_content_data == 1:
    return X, X_content
  return X

      
      


