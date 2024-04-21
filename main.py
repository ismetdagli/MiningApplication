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
import joblib
import pickle
import pywt
import numpy as np
import pandas as pd

#from dataloading.loaders import load_strain_gauge_limestone, load_cap_limestone
from loader import load_audio_files
from windowizer import Windowizer, window_maker
from custom_pipeline_elements import SampleScaler, ChannelScaler, FFTMag, WaveletDecomposition

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
freq_transforms1 = [('FFT_Rt', FFTMag(1, power='SQRT'))]
freq_transforms2 = [("FreqControl2", None)]



classifiers = [
  # ('rbf_svm', svm.SVC(class_weight='balanced')),
              #  ('MLPClass1', MLPClassifier(solver='lbfgs', activation='relu', 
              #   alpha=1e-10, tol=1e-8,
              #   hidden_layer_sizes=(windowed_audio_data[0].shape[0], 
              #                       windowed_audio_data[0].shape[0]), 
              #   max_iter=300, verbose=False)),
               ('MLPClass2', MLPClassifier(solver='lbfgs', activation='relu', 
                alpha=1e-10, tol=1e-8,
                hidden_layer_sizes=(2*windowed_audio_data[0].shape[0], 
                                    2*windowed_audio_data[0].shape[0]),
                max_iter=300, verbose=False))
              #  ('MLPClass3', MLPClassifier(solver='lbfgs', activation='relu', 
              #   alpha=1e-10, tol=1e-8,
              #   hidden_layer_sizes=(2*windowed_audio_data[0].shape[0], 
              #                       2*windowed_audio_data[0].shape[0], 
              #                       windowed_audio_data[0].shape[0]), 
              #   max_iter=300, verbose=False))
              #  ('K5N', KNeighborsClassifier(n_neighbors=5)),
              #  ('K10N', KNeighborsClassifier(n_neighbors=10)),
              #  ('K15N', KNeighborsClassifier(n_neighbors=15))
] 

test=MLPClassifier(solver='lbfgs', activation='relu', 
                alpha=1e-10, tol=1e-8,
                hidden_layer_sizes=(2*windowed_audio_data[0].shape[0], 
                                    2*windowed_audio_data[0].shape[0]),
                max_iter=300, verbose=False)
print("\n ",2*windowed_audio_data[0].shape[0])


# Do experiment, record data to list
# Save results from experiments to list of list of pairs
results_list = []
data_X = windowed_audio_data
data_Y = [wear_classes2ints[label] for label in windowed_audio_labels] 

scorings = ['f1_macro','accuracy']

for ft1 in freq_transforms1:
 for ft2 in freq_transforms2:
  for sc2 in scalings2:
   for cls in classifiers:
      cross_val = ShuffleSplit(n_splits=number_cross_validations, test_size=my_test_size, 
                               random_state = 711711)
      my_pipeline = Pipeline([ft1, ft2, sc2, cls])

      params = "None"
      if (cls[0] == 'rbf_svm'):
        print("Fitting svm hyper param")
        C_range = np.logspace(0, 3, 7)
        gamma_range = np.logspace(-4, -1, 7)
        param_grid = {"rbf_svm__gamma" : gamma_range, 
                          "rbf_svm__C" : C_range,
                      "rbf_svm__class_weight" : ["balanced"]}
        grid = GridSearchCV(my_pipeline, param_grid=param_grid, cv=cross_val, verbose=1, n_jobs=number_parallel_jobs)
        grid.fit(data_X, data_Y)

        print(
             "The best parameters are %s with a score of %0.2f"
             % (grid.best_params_, grid.best_score_)
        )
        params = str(grid.best_params_)        

        my_pipeline.set_params(**grid.best_params_)

      scores = cross_validate(my_pipeline, data_X, data_Y, cv=cross_val,
                                scoring=scorings, n_jobs=number_parallel_jobs)
      # model = my_pipeline.fit(data_X,data_Y)
      # print("cls: ",cls[0])
      # if (cls[0] == 'K5N'):
      #   import dill as pickle
      #   with open('K5N.pkl', 'wb') as fid:
      #       pickle.dump(model, fid)
      # if (cls[0] == 'K10N'):
      #   import dill as pickle
      #   with open('K10N.pkl', 'wb') as fid:
      #       pickle.dump(model, fid)
      # if (cls[0] == 'K15N'):
      #   import dill as pickle
      #   with open('K15N.pkl', 'wb') as fid:
      #       pickle.dump(model, fid)

      model = my_pipeline.fit(data_X,data_Y)
      if (cls[0] == 'MLPClass2'):
        import dill as pickle
        with open('MLPClass3.pkl', 'wb') as fid:
            pickle.dump(model, fid)
      # Concat to data frame
      # dynamic_params_pairs = [("num_samples", [str(len(data_X))]),
      #                         ("sample_lens", [data_X[0].shape[0]]),
      #                         ("freq1", [my_pipeline.steps[0][0]]), 
      #                         ("freq2", [my_pipeline.steps[1][0]]), 
      #                         ("stand2", [my_pipeline.steps[2][0]]),
      #                         ("classifier", [my_pipeline.steps[3][0]]),
      #                         ("params", [params]),
      #                         ("mean_score", [str(scores["test_f1_macro"].mean())]),
      #                         ("std_dev", [str(scores["test_f1_macro"].std())]),
      #                         ("acc", [str(scores["test_accuracy"].mean())]), 
      #                         ("acc_dev", [str(scores["test_accuracy"].std())])]

      # # Create data frame from static and dynamic data, append to results dataframe
      # experiment_data_pairs = static_params_pairs + dynamic_params_pairs
      # results_list.append(experiment_data_pairs)

      # Progress UI
      print(".", end='', flush=True)

that_time = time.time()
print(" Done! Took {0} sec; Saving data...".format(that_time - this_time))

# print list and save to file
#for result in results:
#  print(result)

# # Get list of column names from first entry in results list
# result_columns = [item[0] for item in results_list[0]] 

# # Make list of rows as tuples
# result_rows = []
# for res in results_list:
#   names  = [item[0] for item in res] # Unused
#   values = [item[1][0] for item in res] 
#   result_rows.append(values)

# results_frame = pd.DataFrame(data=result_rows, columns=result_columns)

# print("results frame")
# print(results_frame)

# ## Write file
# os.makedirs('./out', exist_ok=True)
# timestr = time.strftime("%Y%m%d_%H%M%Sresults.csv")

# #with open('./out/' + timestr, 'w') as f:
# #  for line in results:
# #    f.write(','.join(line) + '\n')

# ## Write better file
# outfilename = './out/' + "CONCRETE_" + timestr
# results_frame.to_csv(outfilename, index_label=False, index=False) 

print("Have a nice day!")

# Score pipelines using default SVM with linear kernal
# Iterate through material and wear for both sensors
# and all hyperparams using the chosen window settings
# also test different test train splits