import numpy as np
import tensorflow as tf

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

def data_preprocess(directory, years, months, part_months, zone, new_size, ind_min, ind_max):
  data = np.array([])
  for year in years:
    for month in months:
      for part_month in part_months:
        fname = directory + f'{zone}_reflectivity_old_product_{str(year)}/{zone}_reflectivity_old_product_{str(year)}/reflectivity-old-{zone}-{str(year)}-{str(month).zfill(2)}/reflectivity-old-{zone}-{str(year)}-{str(month).zfill(2)}/reflectivity_old_{zone}_{str(year)}_{str(month).zfill(2)}.{str(part_month)}.npz'
        d = np.load(fname, allow_pickle=True)

        # Check if ind_min and ind_max are defined
        if ind_min != None and ind_max != None:
          data_temp = d['data'][ind_min:ind_max, :, :]
        else:
          data_temp = d['data']

        data_temp = np.expand_dims(data_temp, -1) # add channel dimension for tf handling (channel=1, ie grayscale)

        # Resize the dimensions of the images to new_size
        if len(new_size) > 0:
          data_temp = tf.image.resize(data_temp, new_size)
          data_temp = np.asarray(data_temp)

        # Interpolate frame where values are missing
        if d['miss_dates'].shape[0] != 0:
          data_temp = interpolate_missing_data(data_temp, d['miss_dates'], d['dates'])

        #Store everything in one dataset
        if data.shape[0] == 0:
          data = data_temp
        else:
          data = np.vstack((data, data_temp))
        
        print("Year: {} Month: {} Part of the month: {}, Done !".format(year, month, part_month))
  
  data = data/np.amax(data) # normalize between 0 and 1 (min-max scaling)
  return data

def interpolate_missing_data(data, miss_dates, dates):
  for i in range(miss_dates.shape[0]):
    if miss_dates[i] < dates[0]:
      data = np.insert(data, 0, data[0], axis=0)
    elif miss_dates[i] > dates[-1]:
      data = np.insert(data, -1, data[0], axis=0)
    else:
      m = np.argmin(abs(dates - miss_dates[i]))
      if dates[m] < miss_dates[i]:
        data = np.insert(data, m+1, (data[m] + data[m+1])/2, axis=0)
      else:
        data = np.insert(data, m, (data[m] + data[m-1])/2, axis=0)
  return data

# A function to help get the training and validation data correctly
def multivariate_data(dataset, start_index, end_index, history_size):
  data = []
  labels = []

  if end_index is None:
    end_index = dataset.shape[0]

  for i in range(start_index, end_index-1, history_size + 1):
    indices = range(i-history_size, i)
    indices_labels = range(i-history_size + 1, i+1)
    data.append(dataset[indices])
    labels.append(dataset[indices_labels])

  return np.array(data), np.array(labels)