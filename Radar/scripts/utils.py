import numpy as np
import tensorflow as tf


def get_coords(directory, zone):
  fname_coords = directory + f'radar_coords_{zone}.npz'
  coords = np.load(fname_coords, allow_pickle=True)
  resolution = 0.01 #spatial resolution of radar data (into degrees)
  lat = coords['lats']-resolution/2
  lon = coords['lons']+resolution/2
  return [lat, lon]


def add_new_channel(X, new_channel):
  X_num_dim = len(X.shape)
  new_channel_num_dim = len(new_channel.shape)
  new_channel_temp = new_channel
  for i in range(X_num_dim - new_channel_num_dim - 1, -1, -1):
    new_channel_temp = tf.expand_dims(new_channel_temp, axis=0)
    new_channel_temp = tf.repeat(new_channel_temp, X.shape[i], axis=0)
  new_X = tf.concat([X, new_channel_temp], axis=-1)
  return new_X


def interpolate_missing_data(data, miss_dates, dates):
  for i in range(miss_dates.shape[0]):
    if miss_dates[i] < dates[0]:
      data = np.insert(data, 0, data[0], axis=0)
      dates = np.insert(dates, 0, miss_dates[i])
    elif miss_dates[i] > dates[-1]:
      data = np.insert(data, -1, data[-1], axis=0)
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


def unison_shuffle(a, b):
    perm = np.random.permutation(a.shape[0])
    a_shuffled = tf.gather(a, perm, axis=0)
    b_shuffled = tf.gather(b, perm, axis=0)
    return a_shuffled, b_shuffled


def round_to_day(dt):
    return dt + datetime.timedelta(hours = -dt.hour, minutes = -dt.minute, seconds = -dt.second)


def get_hour(dt):
  return dt.hour


def get_day(dt):
  return dt.day