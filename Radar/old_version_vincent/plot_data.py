import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from matplotlib import colors
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 3

def plot_history(history, normalization_min, save=False, foldername=''):
  if save == True:
    try:
        os.mkdir(foldername)
    except OSError:

        print("Creation of the directory %s failed" % foldername)
    else:
        print("Successfully created the directory %s " % foldername)

  # summarize history for accuracy
  plt.figure(figsize=(12,6))
  plt.plot(history.history['acc'], label='Train')
  plt.plot(history.history['val_acc'], label='Val')
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(loc='upper left')
  plt.show()
  if save == True:
    plt.savefig("".join([foldername,'accuracy.png']))
  # files.download('accuracy.png')

  # summarize history for loss
  plt.figure(figsize=(12,6))
  plt.plot(history.history['loss'], label='Train')
  plt.plot(history.history['val_loss'], label='Val')
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(loc='upper left')
  plt.show()
  if save == True:
    plt.savefig("".join([foldername,'loss.png']))
  # files.download('loss.png')

  if normalization_min == 0:
    plt.figure(figsize=(12,6))
    plt.plot(history.history['cor'], label='Train')
    plt.plot(history.history['val_cor'], label='Val')
    plt.title('Model Correlation')
    plt.ylabel('Correlation')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()
    if save == True:
      plt.savefig("".join([foldername,'cor.png']))
    # files.download('accuracy.png')

    # summarize history for loss
    plt.figure(figsize=(12,6))
    plt.plot(history.history['prec'], label='Train')
    plt.plot(history.history['val_prec'], label='Val')
    plt.title('Model precision')
    plt.ylabel('precision')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()
    if save == True:
      plt.savefig("".join([foldername,'prec.png']))
    # files.download('loss.png')

    plt.figure(figsize=(12,6))
    plt.plot(history.history['recall'], label='Train')
    plt.plot(history.history['val_recall'], label='Val')
    plt.title('Model recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()
    if save == True:
      plt.savefig("".join([foldername,'recall.png']))
    # files.download('accuracy.png')

def plot_track(true_track, track, normalization_min, threshold_value, new_size, output_timeframes, save=False, foldername=''):
  if save == True:
    try:
        os.mkdir(foldername)
    except OSError:

        print("Creation of the directory %s failed" % foldername)
    else:
        print("Successfully created the directory %s " % foldername)

  cmap = colors.ListedColormap(['silver','white', 'darkslateblue', 'mediumblue','dodgerblue', 
                              'skyblue','olive','mediumseagreen','cyan','lime','yellow',
                              'khaki','burlywood','orange','brown','pink','red','plum'])
  bounds = [-1,0,2,4,6,8,10,15,20,25,30,35,40,45,50,55,60,65,75]
  norm = colors.BoundaryNorm(bounds, cmap.N)
  if normalization_min == -1:
    track = track/2 + 0.5
    true_track = true_track/2 + 0.5
  track = track*threshold_value
  true_track = true_track*threshold_value
  for i in range(1+output_timeframes):
      plt.figure(figsize=(10, 5))
      plt.subplot(121)
      if i >= 1:
          plt.text(1, 3, 'Prediction', color='w')
      else:
          plt.text(1, 3, 'Initial trajectory', color='w')
      plt.pcolormesh(range(new_size[0]), range(new_size[1],-1,-1), track[0, i, :, :, 0],cmap=cmap, norm=norm)
      plt.subplot(122)
      plt.text(1, 3, 'Ground truth', color='w')
      plt.pcolormesh(range(new_size[0]), range(new_size[1],-1,-1),true_track[i, :, :, 0],cmap=cmap, norm=norm)
      cbar = plt.colorbar(cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, 
                orientation= 'vertical').set_label('Rainfall (in 1/100 mm) / -1 : missing values')
      if save == True:
          plt.savefig('train_%i.png' % (i+1))
      #files.download('train_%i.png' % (i+1))
