import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import pdb
import itertools 


sample_file_name = '20220620_094439results.csv' # Representative sample 0.25 test ratio, 0.2s window size
#sample_file_name = '20220629_104255results.csv' # 0.5 test ratio, 0.2s window size
#sample_file_name = '20220629_105237results.csv' # 0.75 test ratio, 0.2s window size

print_z_scores = True

# Group by application > frequency technique > normalization scheme

data_bank = pd.read_csv('./data/' + sample_file_name)

applications = data_bank.groupby("application")

normindex = {("ScaleControl1", "ScaleControl2"): 0, ("FeatureScaler1", "ScaleControl2"): 1,
             ("FeatureScaler1", "FeatureScaler2"):2, ("ScaleControl1", "FeatureScaler2"):3}
norm_colors_by_index = {0: "red", 1: "orange", 2: "blue", 3: "green"}


freqindex = {"FFT_Mag": 0, "FFT_MagSq": 1, "FFT_MagRt": 2, "FreqControl": 3}
appindex  = {"cap mat.": 0, "sg mat.": 1, "cap wear": 2, "sg wear": 3}

app_spacing = 50
freq_spacing = 10
norm_spacing = 1.5
basis = [norm_spacing, freq_spacing, app_spacing]

x_plot_points = []
y_plot_points = []
y_plot_errors = []
c_plot_colors = []

# Pair coordinates with hierarchy of data
for app_name, app_group in applications:
  freq_techs = app_group.groupby("fft")
  for freq_name, freq_group in freq_techs:
    for _, row in freq_group.iterrows():
      coeff = [normindex[(row["stand1"], row["stand2"])], freqindex[freq_name], appindex[app_name]]
      x_plot_points.append(sum([c*b for c,b in zip(coeff, basis)]))
      c_plot_colors.append(norm_colors_by_index[coeff[0]])
      y_plot_points.append(row["mean_score"])
      y_plot_errors.append(row["std_dev"]/2.0) # plot is +/- this value

# Calculate Z scores for methods in application
for app_name, app_group in applications:
  print("Best methods are: \n{0}".format(app_group[app_group["mean_score"] == app_group["mean_score"].max()]))
  if print_z_scores:
    z_scores = {}
    for combo in itertools.permutations(app_group.iterrows(),2): # do permutations for easy reading
      z_scores[((combo[0][1]["stand1"], combo[0][1]["stand2"], combo[0][1]["fft"]),
                (combo[1][1]["stand1"], combo[1][1]["stand2"], combo[1][1]["fft"]))] = (combo[0][1]["mean_score"] - combo[1][1]["mean_score"]) / np.sqrt(
                                                                                        combo[0][1]["std_dev"]*combo[0][1]["std_dev"]/combo[0][1]["num_splits"] + 
                                                                                        combo[1][1]["std_dev"]*combo[1][1]["std_dev"]/combo[1][1]["num_splits"] ) 
    print("Z-scores for {0}".format(app_name))
    [print(item) for item in z_scores.items()]

fig1, ax = plt.subplots()

plt.bar(x=x_plot_points, height=y_plot_points, yerr = y_plot_errors, color=c_plot_colors, capsize=3, ecolor='gray')

#labels the y axis
yname = "F1 scores with 1 SD error bars"
yfont = {'family': 'Sans','color':  'k','weight': 'normal','size': 15,}
plt.ylabel(yname, fontdict=yfont, labelpad=15)
#the title
titlename = "Performance of methods with {0}s window width, {1} test ratio".format(
              data_bank.iloc[0]["window_duration"], data_bank.iloc[0]["test_ratio"])
labelfont = {'family': 'Sans','color':  'k','weight': 'bold','size': 15,}
plt.title(titlename,fontdict=labelfont, pad=25)
#labels the x axis
xname = "Methods"
xfont= {'family':'Sans', 'color':'k','weight':'normal','size': 12,}
plt.xlabel(xname, fontdict=xfont, labelpad=10)

#group names of applications
plt.tick_params(bottom=True, top=False, left=True, right=False)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
fft_methods = 4 * ["FFT_Mag", "FFT_MagSq", "FFT_MagRt", "FreqControl"]
pos = []
for apps in range(4):
  for freqs in range(4):
    pos.append(freqs * freq_spacing + apps * app_spacing + 1.5*norm_spacing)

plt.xticks(pos, fft_methods, rotation=-45)

#labeling  application

#cap mat
left, bottom, width, height = (-2, 0, 38, 0.05)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        facecolor = 'white',
                        edgecolor = 'black',
                       linewidth=1,
                       zorder=2,
                       )
plt.gca().add_patch(rect)
# add text with text() function in matplotlib
plt.text(8, 0.015,'cap mat',fontsize=16, color="black", weight="bold")

#sg mat
left, bottom, width, height = (48, 0, 38, 0.05)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        facecolor = 'white',
                        edgecolor = 'black',
                       linewidth=1,
                       zorder=2,
                       )
plt.gca().add_patch(rect)
# add text with text() function in matplotlib
plt.text(59, 0.015,'sg mat',fontsize=16, color="black", weight="bold")

#cap wear
left, bottom, width, height = (98, 0, 38, 0.05)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        facecolor = 'white',
                        edgecolor = 'black',
                       linewidth=1,
                       zorder=2,
                       )
plt.gca().add_patch(rect)
# add text with text() function in matplotlib
plt.text(107.5, 0.015,'cap wear',fontsize=16, color="black", weight="bold")

#sg wear
left, bottom, width, height = (148, 0, 38, 0.05)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        facecolor = 'white',
                        edgecolor = 'black',
                       linewidth=1,
                       zorder=2,
                       )
plt.gca().add_patch(rect)
# add text with text() function in matplotlib
plt.text(159, 0.015,'sg wear',fontsize=16, color="black", weight="bold")

#legend of the colors of the bars
colors = { 'None':'red','Before':'orange','Before + After':'blue','After':'green'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
locationlegend = 'upper left'
#lables the legend
titlelegend =  'FFT Normalization'
plt.legend(handles, labels,bbox_to_anchor=(1.02, 1), loc= locationlegend, borderaxespad=0,title =titlelegend)




plt.show(block=True)

