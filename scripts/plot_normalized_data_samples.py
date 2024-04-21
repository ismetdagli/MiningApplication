# This file plots the data used in main.py after it has been normallized
# and for each pre-processing technique and the time domain control.


import pdb
import time
import os
import argparse
import decimal

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
from sklearn.metrics import confusion_matrix, f1_score, make_scorer

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

import pywt
import numpy as np
import pandas as pd

#from dataloading.loaders import load_strain_gauge_limestone, load_cap_limestone
from loader import load_audio_files
from windowizer import Windowizer, window_maker
from custom_pipeline_elements import SampleScaler, ChannelScaler, FFTMag, WaveletDecomposition

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#help words
shape_options = "hamming,boxcar"
duration_options = "0 - 10 second duration"
overlap_options = "overlap ratio 0-1"
#required inputs
allowed_overlap = [x/100 for x in range(0, 101, 5)]

## Start Simulation Parameters ##

name = "Concrete Tool Wear"

# Computation parameter
number_parallel_jobs = 3

#default values
window_shape    = "hamming" #"boxcar" # from scipy.signal.windows
window_duration = 0.2 # seconds
window_overlap  = 0.5 # ratio of overlap [0,1)

# Machine learning sampling hyperparameters #
number_cross_validations = 8
my_test_size = 0.5

# Load data
audio_fs = 44100 # Samples per second for each channel
downsample_factor = 8

print("Loading data...")
this_time = time.time()

# Load and Downsample, adjust audio_fs
audio_fs = int(audio_fs/downsample_factor)
raw_audio_data, metadata = load_audio_files("./raw_audio/classifications.txt", integer_downsample=downsample_factor)


## Allow command line overrides
# Making command line argument for window shape
parser = argparse.ArgumentParser()
parser.add_argument("--window_shape", default=window_shape, type=str,
  help=shape_options)
# Making command line argument for window duration
parser.add_argument("--window_duration", default=window_duration, type=float,
  help=duration_options)
# Making command line argument for window overlap
parser.add_argument("--window_overlap", type=float, default=window_overlap,
  help=overlap_options)

args = parser.parse_args()
# Making the overlap between 0-1
if args.window_overlap > 1:
  raise Exception("Sorry, no numbers above 1")
else:
  pass
if args.window_overlap < 0:
  raise Exception("Sorry, no numbers below zero") 
else:
  pass
# Printing what the values are
if args.window_shape:
    print("window shape is",args.window_shape)
if args.window_duration:
    print("window duration is",args.window_duration)
if args.window_overlap:
    print("window overlap is",args.window_overlap)
else: 
      print("windows don't overlap")

window_len = int(args.window_duration*audio_fs)
## End command line parsing

static_params_pairs = [ ("name", [name]),
                        ("window_shape", [args.window_shape]),
                        ("window_duration", [args.window_duration]),
                        ("window_overlap", [args.window_overlap]),
                        ("window_len", [window_len]),
                        ("number_parallel_jobs", [number_parallel_jobs]),
                        ("number_cross_validations", [number_cross_validations]),
                        ("my_test_size", [my_test_size]),
                        ("audio_fs", [audio_fs]),
                        ("downsample_factor", [downsample_factor]),
                        ("load_date_time", [this_time]) ] # All parameters 

## End default parameters and loading ##
## End parameters ##

# Apply windowing

audio_window = Windowizer(window_maker(args.window_shape, int(args.window_duration*audio_fs)), args.window_overlap)
windowed_audio_data, windowed_audio_labels = audio_window.windowize(raw_audio_data, metadata)
      

wear_list = ["New", "Moderate", "Worn"]
wear_classes2ints = {"New":0, "Moderate":1, "Worn":2}
wear_ints2classes = {v: k for k,v in wear_classes2ints.items()}

# Build preprocessing lists for pipeline
# scale1: [std, samp, chan, none]
# freq_transform: [abs(rfft()), abs(rfft()).^2, sqrt(abs(rfft())), none]
# scale2: [std, samp, chan, none]

that_time = time.time()
print("Data loaded in {0} sec; performing experiments".format(that_time - this_time),
      end='', flush=True)
this_time = time.time()
# Build pipeline
#scalings1 = [("ScaleControl1", None)] # ("FeatureScaler1", StandardScaler())
scalings2 = [("FeatureScaler2", StandardScaler())] #, ("ScaleControl2", None)]
freq_transforms1 = [('FFT_Rt', FFTMag(1, power='SQRT')),
                    ('FFT_Mag', FFTMag(1)),
                    ('FFT_Sq', FFTMag(1, "SQUARE")),
                    ("FreqControl1", None)]

# Do experiment, record data to list
# Save results from experiments to list of list of pairs
results_list = []
data_X = windowed_audio_data
data_Y = [wear_classes2ints[label] for label in windowed_audio_labels] 

# Get Worn data, Mod data, New data
labelled_data = {label:[] for label in windowed_audio_labels}
for x,y in zip (data_X, data_Y):
  labelled_data[wear_ints2classes[y]].append(x)

transformed_data = {label:{} for label in windowed_audio_labels}

for ft1 in freq_transforms1:
  for sc2 in scalings2:
    for wear in wear_list:
      my_pipeline = Pipeline([ft1, sc2])

      my_pipeline.fit(labelled_data[wear])
      transformed_data[wear][ft1[0]] = my_pipeline.transform(labelled_data[wear])

      # Progress UI
      print(".", end='', flush=True)

that_time = time.time()
print(" Done! Took {0} sec; Saving data...".format(that_time - this_time))
print(transformed_data)

# Compute distributions
# Plot for each wear type and technique
# Sort data by wear level
# Find average and variance of each wear level and processing technique
names = []
values_list = []
for ft1 in freq_transforms1:
  for wear in wear_list:
    names.append(f"{wear} with {ft1[0]}")
    values_list.append(np.array(transformed_data[wear][ft1[0]]))

avgs = {}
devs = {}
for val, name in zip(values_list, names):
  avgs[name] = np.mean(val, axis=1)
  devs[name] = np.std(val, axis=1)

print(names)
print(avgs)

# Plot distribution vectors for each wear level for frequency
# Set figure sizes
fontsize = 18
plt.rc('font', size=fontsize, family='sans')
plt.rc('axes', titlesize=fontsize)
plt.rc('axes', labelsize=fontsize)
plt.rc('legend', fontsize=fontsize)
plot_width = 3.6 # pt

for idx,ft1 in enumerate(freq_transforms1):
  fig, axs = plt.subplots(len(wear_list))

  for index in range(len(wear_list)):
    # Plot domain
    print(f"Plotting {names[index]}")
    domain_vals = np.linspace(0, audio_fs/2, len(avgs[names[index]]))
    if "Control" in names[index]:
      domain_vals = np.linspace(0, window_duration, len(avgs[names[index]]))
      print("DOMAIN CONTROL!")

    namedex = idx * len(wear_list) + index
    axs[index].fill_between(domain_vals,
      avgs[names[namedex]] + devs[names[namedex]],
      avgs[names[namedex]] - devs[names[namedex]],
      alpha=0.5, color='tab:purple')
    axs[index].plot(domain_vals, avgs[names[namedex]],
    color='tab:green', linewidth=plot_width)
    axs[index].set_title(names[namedex])
  
  fig.show()
    #axs[index][0].set_ylim(-6000, 6000)
    ##axs[index][0].legend(["Mean", r"$\pm$ 1 Std. Dev."], loc="upper right")
    #axs[index][0].set_xlabel("Time (s)")
    #axs[index][0].set_ylabel("Amplitude")

#  # Plot freq domain
#  axs[index][1].fill_between(freq_vals,
#    avgs[names[index+3]] + devs[names[index+3]],
#    avgs[names[index+3]] - devs[names[index+3]],
#    alpha=0.5, color='tab:purple')
#  axs[index][1].fill_between([50, 250], 0, 500000, color='tab:red', alpha=0.2)
#  axs[index][1].fill_between([320, 450], 0, 500000, color='tab:red', alpha=0.2)
#  axs[index][1].fill_between([750, 850], 0, 500000, color='tab:red', alpha=0.2)
#  axs[index][1].fill_between([1080, 1110], 0, 500000, color='tab:red', alpha=0.2)
#  axs[index][1].plot(freq_vals, avgs[names[index+3]],
#  color='tab:green', linewidth=plot_width)
#  axs[index][1].set_title(names[index+3])
#  axs[index][1].set_ylim(0, 3.5e5)
#  axs[index][1].legend(["Mean", r"$\pm$ 1 Std. Dev."], loc="upper right")
#  axs[index][1].set_xlabel("Freq. (Hz)")
#  axs[index][1].set_ylabel("Spectra Magnitude")
#
#
plt.show(block=False)
input("Press Enter to close...")



