
# coding: utf-8

# In[1]:

# Running on GPU?

#import setGPU


# In[2]:

import math
import numpy as np
import pandas as pd
import cPickle as pickle
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable

import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Reshape, Conv1D, MaxPooling1D, AveragePooling1D, UpSampling1D, InputLayer
from scipy import ndimage, misc

from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils import class_weight


# # Local approach to Drift Tubes Digi Occupancy Anomaly Problem

# * [Load data](#Load-occupancy-data-to-the-dataframe)
# * [Preprocessing](#Preprocessing)
#     * [Visualize preprocessing](#Visualize-preprocessing-steps)
# * [Anomaly detection](#Searching-for-anomalies)
#     * [Production baseline](#Production-baseline)
#     * [Dataset split](#Split-the-dataset)
#     * [Simple statistics](#Benchmarking-statistical-and-filter-tests)
#     * [One Class SVM & Isolation Forst](#Benchmarking-SVM-and-IF)
#     * [Neural Networks](#Benchmarking-neural-networks)
# * [Stability](#Checking-stability)

# In[3]:

# Load models from disk

#LOAD_MODELS = True


# In[4]:

# Set a random seed to reproduce the results

rng = np.random.RandomState(0)


# In[5]:

# Change presentation settings

#get_ipython().magic('matplotlib inline')

matplotlib.rcParams["figure.figsize"] = (8.0, 5.0)

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Liberation Sans']

matplotlib.rcParams["axes.spines.left"] = True
matplotlib.rcParams["axes.spines.top"] = True
matplotlib.rcParams["axes.spines.right"] = True
matplotlib.rcParams["axes.spines.bottom"] = True
matplotlib.rcParams["axes.labelsize"] = 18
matplotlib.rcParams["axes.titlesize"] = 14

#matplotlib.rcParams["xtick.top"] = True
#matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["xtick.labelsize"] = 18
matplotlib.rcParams["ytick.labelsize"] = 18
matplotlib.rcParams["xtick.major.size"] = 10
matplotlib.rcParams["ytick.major.size"] = 10
matplotlib.rcParams["xtick.minor.size"] = 5
matplotlib.rcParams["ytick.minor.size"] = 5
matplotlib.rcParams["xtick.minor.visible"] = True

matplotlib.rcParams["lines.linewidth"] = 2

matplotlib.rcParams["legend.fontsize"] = 14

color_palette = {"Indigo": {
                    50: "#E8EAF6",
                    100: "#C5CAE9",
                    200: "#9FA8DA",
                    300: "#7986CB",
                    400: "#5C6BC0",
                    500: "#3F51B5",
                    600: "#3949AB",
                    700: "#303F9F",
                    800: "#283593",
                    900: "#1A237E"},
                 "Orange": {      
                    50: "#FFF3E0",
                    100: "#FFE0B2",
                    200: "#FFCC80",
                    300: "#FFB74D",
                    400: "#FFA726",
                    500: "#FF9800",
                    600: "#FB8C00",
                    700: "#F57C00",
                    800: "#EF6C00",
                    900: "#E65100"}
                }

line_styles = [(0, ()), (0, (1, 2)), (0, (3, 2))]


# In[6]:

# Set data directory
# You can copy the data from /eos/cms/store/cmst3/group/dehep/adpol/dt/data

data_directory = "../data"
labels_directory = "../data"
models_directory = "../models-local"
figure_export_directory = "../figures"


# ## Load occupancy data to the dataframe

# In[7]:

# Load occupancy data to the dataframe
#distis = [5,20500] 
#distis = [5,50,20300,1717,1718] 
distis = [8010,8050,8100, #faulty like real
          10300,               #hot fake mean=300
          7100,                # all faulty
          1710,1711,1712,1713,1714, #real
          9010,9020,9050,9070,9100,9200] #fake like real

test_distis = [1710,1711,1712,1713,1714]
train_distis = [r for r in distis if r not in test_distis]

gem_vfats = pd.DataFrame()
for disti in distis:
    #print("Loading %s" % run)
    path = "%s/%s.json" % (data_directory, disti)
    gem_vfats = gem_vfats.append(pd.read_json(path),
                                     ignore_index=True);
print("Done. Collected %s vfats" % gem_vfats.shape[0]),


# In[8]:

# Transform the string of hits to list

gem_vfats["content"] = gem_vfats["content"].apply(eval)
gem_vfats["content"] = gem_vfats["content"].apply(np.array)


# In[9]:

# Append score from labels data file

labels_frame = pd.read_csv(("%s/labels.csv" % labels_directory),
                           names=["run",
                                  "disti",
                                  "vfat",
                                  "score"])
gem_vfats = pd.merge(gem_vfats,
                     labels_frame,
                     how="left",
                     on=["disti",
                         "run",
                         "vfat"])
#print labels_frame.score
# ## Preprocessing

# In[10]:

# Use median polling to remove channels with extreme values (smoothing)

SMOOTH_FILTER_SIZE = 3

def median_polling(vfat):
    """Smooths vfat occupancy using median filter"""
    smooth_vfat = []
    for index in range(len(vfat) - (SMOOTH_FILTER_SIZE-1)):
        median = np.median(vfat[ index : index + SMOOTH_FILTER_SIZE ])
        smooth_vfat.append(median)
    return np.array(smooth_vfat)

gem_vfats["content_smoothed"] = gem_vfats["content"].apply(median_polling)


# In[11]:

print("Minimum raw length: % s" % min(gem_vfats["content"].apply(len)))
print("Maximum raw length: % s" % max(gem_vfats["content"].apply(len)))
print("Minimum smoothed length: % s" % min(gem_vfats["content_smoothed"].apply(len)))
print("Maximum smoothed length: % s" % max(gem_vfats["content_smoothed"].apply(len)))


# In[12]:

# Use linear interpolation to resize all the data samples (standardization)

def resize_occupancy(vfat):
    """Resizes occupancy to a given size using bilinear interpolation"""
    return misc.imresize(np.array(vfat).reshape(1, -1),( 1, SAMPLE_SIZE), interp="bilinear", mode="F").reshape(-1)

SAMPLE_SIZE = min(gem_vfats["content"].apply(len))
gem_vfats["content_resized"] = gem_vfats["content"].apply(resize_occupancy)

SAMPLE_SIZE = min(gem_vfats["content_smoothed"].apply(len))
gem_vfats["content_smoothed_resized"] = gem_vfats["content_smoothed"].apply(resize_occupancy)


# In[13]:

# Normalize the data (normalization)

def scale_occupancy(vfat):
    """Scales vfat data using MaxAbsScaler"""
    # Need to reshape since scaler works per column
    vfat = vfat.reshape(-1, 1)
    scaler = MaxAbsScaler().fit(vfat)
    return scaler.transform(vfat).reshape(1, -1)

gem_vfats["content_scaled"] = gem_vfats["content_resized"].apply(scale_occupancy)
gem_vfats["content_smoothed_scaled"] = gem_vfats["content_smoothed_resized"].apply(scale_occupancy)


# ### Visualize preprocessing steps

# In[14]:

# Combine per vfat data to per chamber data

gem_chambers = pd.DataFrame()
for disti in distis:
    for run in range(0,100):
        if labels_frame[(labels_frame.disti == disti) &
                        (labels_frame.run == run)].empty:
            continue
        chamber = gem_vfats[
            (gem_vfats.disti == disti) &
            (gem_vfats.run == run)].sort_values("vfat",
                                                ascending=1)
        if not len(chamber):
            continue
        occupancy_raw = [vfat.tolist() for vfat in chamber["content"]]
        occupancy_smoothed = [vfat.tolist() for vfat in chamber["content_smoothed"]]

        occupancy_resized = np.concatenate(
            chamber["content_resized"].values).reshape(-1, 32)
        occupancy_scaled = np.concatenate(
            chamber["content_scaled"].values).reshape(-1, 32)
        occupancy_smoothed_resized = np.concatenate(
            chamber["content_smoothed_resized"].values).reshape(-1, 30)
        occupancy_smoothed_scaled = np.concatenate(
            chamber["content_smoothed_scaled"].values).reshape(-1, 30)

        extended_size_smoothed = max((len(_) for _ in occupancy_smoothed))
        extended_size_raw = max((len(_) for _ in occupancy_raw))
        
        for index, vfat in enumerate(occupancy_raw):
            vfat.extend([np.nan]*(extended_size_raw-len(vfat)))
            occupancy_raw[index] = vfat

        for index, vfat in enumerate(occupancy_smoothed):
            vfat.extend([np.nan]*(extended_size_smoothed-len(vfat)))
            occupancy_smoothed[index] = vfat

        score = sum(chamber.score.values)
        data = {"disti": disti,
                "run" : run,
                "score": score,
                "content_raw": np.concatenate(
                    occupancy_raw).reshape(len(chamber), extended_size_raw),
                "content_smoothed": np.concatenate(
                    occupancy_smoothed).reshape(len(chamber), extended_size_smoothed),
                "content_resized": occupancy_resized,
                "content_smoothed_resized": occupancy_smoothed_resized,
                "content_scaled": occupancy_scaled,
                "content_smoothed_scaled": occupancy_smoothed_scaled}
        
        gem_chambers = gem_chambers.append(pd.Series(data),
                                           ignore_index=True)

# In[15]:

# Plotting utils

def plot_occupancy_hitmap(data, title, save_name, unit):
    """Visualizes occupancy hitmap"""
    fig, ax = plt.subplots()
    
    ax = plt.gca()
    
    ax.set_xlim([-2, data.shape[1]+1])
    ax.set_yticklabels(["1", "9", "17"])
    ax.set_yticks([0, 8, 16])
    ax.set_ylim([25,-2])

    plt.xlabel("Channel", horizontalalignment='right', x=1.0)
    plt.ylabel("Layer", horizontalalignment='right', y=1.0)
    
    # Deal with .eps export
    masked_array = np.ma.array (data, mask=np.isnan(data))
    cmap = matplotlib.cm.viridis
    cmap.set_bad("white", 1.)
    
    im = ax.imshow(data, interpolation="nearest", cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    if unit == "a.u.":
        form = '%.2f'
    else:
        form = '%.0f'
    
    plt.colorbar(im,
                 cax=cax,
                 format=form,
                 ticks=[np.min(np.nan_to_num(data)),
                        np.max(np.nan_to_num(data))])
    
    plt.title(title, loc="right")   
    
    ax.text(1.1, 0.75, unit, rotation=90,
        verticalalignment="top", horizontalalignment="right",
        transform=ax.transAxes, color="black", fontsize=16)
    
    ax.text(0, 1.16, "CMS", weight='bold',
        verticalalignment="top", horizontalalignment="left",
        transform=ax.transAxes, color="black", fontsize=18)

    if save_name:
        fig.savefig("%s/occupancy%s.eps" % (figure_export_directory,
                                           save_name),
                    bbox_inches="tight")
    #plt.show();
    
def get_title(title, show):
    """Generates title for occupancy plot"""
    return ("%sRun: %s, M: %s" % 
            (title, int(show.disti), show.run))

def visualize_preprocessing(show, smoothed):
    """Visualizes preprocessing steps"""
    if smoothed:
        plot_occupancy_hitmap(show.content_smoothed,
                              get_title("Smoothed Occupancy, ", show), "F", "a.u.")
        plot_occupancy_hitmap(show.content_smoothed_resized,
                              get_title("Standardized Occupancy, ", show), "G", "a.u.")
        plot_occupancy_hitmap(show.content_smoothed_scaled,
                              get_title("Scaled Occupancy, ", show), False, "a.u.") 

    else:
        plot_occupancy_hitmap(show.content_raw,
                              get_title("Raw Occupancy, ", show), "D", "counts")
        plot_occupancy_hitmap(show.content_resized,
                              get_title("Standardized Occupancy, ", show), "E", "a.u.")
        plot_occupancy_hitmap(show.content_scaled,
                              get_title("Scaled Occupancy, ", show), False, "a.u.")


# Example of preprocessing pipeline for <b>chamber without problems</b>:

# In[16]:

dt = gem_chambers[gem_chambers.score == 24].iloc[4]
plot_occupancy_hitmap(dt.content_raw, get_title("", dt), "A", "counts")

# Example of preprocessing pipeline for <b>chamber with one faulty vfat</b>:

# In[17]:

dt = gem_chambers[gem_chambers.score == 23].iloc[0]
plot_occupancy_hitmap(dt.content_raw, get_title("", dt), "B", "counts")


# Example of preprocessing pipeline for <b>chamber with twelve faulty vfat</b>:

# In[18]:

dt = gem_chambers[gem_chambers.score == 0].iloc[0]
plot_occupancy_hitmap(dt.content_raw, get_title("", dt), "C", "counts")

# Example of <b>alternative</b> preprocessing (with median polling) pipeline for <b>chamber without problems</b>:

# In[19]:

visualize_preprocessing(gem_chambers[gem_chambers.score == 24].iloc[4], False)
visualize_preprocessing(gem_chambers[gem_chambers.score == 24].iloc[4], True)


# ## Searching for anomalies

# In[20]:

# Define styles for plots

lines = [(color_palette["Orange"][600], line_styles[0]),
         (color_palette["Indigo"][800], line_styles[0]),
         (color_palette["Indigo"][200], line_styles[1]),
         (color_palette["Indigo"][200], line_styles[0]),
         (color_palette["Indigo"][500], line_styles[0]),
         (color_palette["Indigo"][500], line_styles[1]),
         (color_palette["Indigo"][100], line_styles[0])]


# In[21]:

# ROC Curve plotting function

def get_roc_curve(test_df, models, working_point=None):
    """Generates ROC Curves for a given array"""
    fig, ax = plt.subplots()
  
    for i, (legend_label, model_score) in enumerate(models):
        fpr, tpr, _ = roc_curve(test_df["score"], test_df[model_score])
        auc_v = round(auc(fpr, tpr), 3)
        plt.plot(fpr,
                 tpr,
                 color=lines[i][0],
                 linestyle=lines[i][1],
                 label=("%s, AUC: %s" % (legend_label, auc_v)))
     #   print legend_label, ":", _
        
    if working_point:
        plt.plot(1-working_point[0],
                 working_point[1],
                 "o",
                 color=color_palette["Orange"][900],
                 markersize=6,
                 label="CNN working point")
    
    plt.legend(frameon=False)
    plt.ylabel("Sensitivity (TPR)", horizontalalignment='right', y=1.0)
    plt.xlabel("Fall-out (1-TNR)", horizontalalignment='right', x=1.0)
    plt.xlim(0,1)
    
    fig.savefig("%s/local_roc.eps" % figure_export_directory, bbox_inches="tight")
    #plt.show();



# In[22]:

# Calculate TPR and TNR

def benchmark(y_true, y_pred):
    """Retrun TPR and TNR"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()   
    sensitivity = float(tp)/(tp+fn)
    specificity = float(tn)/(tn+fp)
    print ("tn:%f ,fp: %f, fn:%f, tp:%f" % (tn,fp,fn,tp))
    print("Model accuracy: %s" % round(accuracy_score(y_true, y_pred), 2))
    print("Model sensitivity: %s" % round(sensitivity, 2))
    print("Model specificity: %s" % round(specificity, 2))
    
    return specificity, sensitivity


# ### Production baseline

# In[23]:

# Yield production algorithm results on the full labelled dataset

def production_test(content):
    """Calculate score that mirrors the test currently in production"""
    return (float(len(np.where(content == 0)[0])) / len(np.where(~np.isnan(content))[0]))

def new_test(content):
    return len(np.where((content > np.mean(content) * 50))[0])
    #return (float(len(np.where((content > np.mean(content) * 50) | (content == 0))[0])) / len(np.where(~np.isnan(content))[0]))
gem_chambers["treshold1"] = gem_chambers["content_raw"].apply(production_test)
gem_chambers["treshold2"] = gem_chambers["content_raw"].apply(new_test)
benchmark(gem_chambers["score"] < 24, (gem_chambers["treshold1"] > 0.4) | (gem_chambers["treshold2"] > 0));


# ### Split the dataset

# In[24]:

def change_score(score):
    return -(2*score-1)

# Change the score (GOOD: 0, BAD: 1)
gem_vfats["score"] = gem_vfats["score"].apply(change_score)

# Get only labelled samples
gem_vfats_scored = gem_vfats[~np.isnan(gem_vfats.score)]
anomalies = gem_vfats_scored[gem_vfats_scored.score == 1]
normalies = gem_vfats_scored[gem_vfats_scored.score == -1]
print("%s faults and %s good samples. In total: %s." %
      (len(anomalies), len(normalies), len(anomalies) + len(normalies)))

# Split the labelled dataset
#anomalies_train, anomalies_test = train_test_split(anomalies, test_size = 0.2, random_state=rng)
#normalies_train, normalies_test = train_test_split(normalies, test_size = 0.2, random_state=rng)
anomalies_train = anomalies[anomalies["disti"].isin(train_distis)]
anomalies_test = anomalies[anomalies["disti"].isin(test_distis)]
normalies_train = normalies[normalies["disti"].isin(train_distis)]
normalies_test = normalies[normalies["disti"].isin(test_distis)]
# Get startified split for neural network validation
neural_anomalies_train, neural_anomalies_val = train_test_split(anomalies_train,
                                                                test_size = 0.2,
                                                                random_state=rng)
neural_normalies_train, neural_normalies_val = train_test_split(normalies_train,
                                                                test_size = 0.2,
                                                                random_state=rng)

# Prepare set for training SVM, IF etc....
vfats_train = pd.concat([anomalies_train, normalies_train])
vfats_test = pd.concat([anomalies_test, normalies_test])

# Prepare training/val sets for neural networks
neural_train = pd.concat([neural_anomalies_train, neural_normalies_train])
neural_val = pd.concat([neural_anomalies_val, neural_normalies_val])

print("Number of anomalies in the train set: %s" % len(anomalies_test))
print("Number of normal in the train set: %s" % len(normalies_test))


# ### Benchmarking statistical and filter tests

# In[25]:

def sobel(content):
    return max(abs(ndimage.sobel(content)[0]))

def variance(content):
    return np.var(content)
    
vfats_test["sobel_score"] = vfats_test["content_smoothed_scaled"].apply(sobel)
vfats_test["variance_score"] = vfats_test["content_smoothed_scaled"].apply(variance)
print disti


get_roc_curve(vfats_test, [("Variance", "variance_score"),
                            ("Sobel", "sobel_score")])


# ### Benchmarking SVM and IF

# In[26]:

# Cross validate model selection using Stratified5Fold and GridSearchCV

def cross_validation_spit(train_X, train_y, clf_i, param_grid, return_params=False):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)
    clf = GridSearchCV(clf_i, param_grid, cv=skf, scoring="roc_auc"); 
    clf.fit(train_X, train_y)
    if return_params:
        return clf.best_params_
    return clf.best_estimator_


# In[28]:

# Training SVM, this may take some time...

#if LOAD_MODELS
#    svmclf = joblib.load("%s/svm.pkl" % models_directory) 
#else:
param_grid = [{"nu": np.array(range(1, 10, 1))/10.0,
               "gamma": [0.1, 0.01, 0.001, 0.0001],
               "kernel": ["linear", "rbf"]}]

svmparams = cross_validation_spit(np.concatenate(
    vfats_train["content_smoothed_scaled"].values),
                                  -vfats_train["score"],
                                  svm.OneClassSVM(random_state=rng),
                                  param_grid)

# Retrain SVM using only good samples
svmclf = svm.OneClassSVM(nu=svmparams.nu,
                         gamma=svmparams.gamma,
                         kernel=svmparams.kernel,
                         random_state=rng)

svmclf.fit(np.concatenate(
    normalies_train["content_smoothed_scaled"].values));
    


# In[29]:

# Training IF, this may take some time...

param_grid = [{"max_samples": [100, 1000],
               "n_estimators": [10, 100],
               "contamination": np.array(range(0, 10, 1)) / 100.0}]

ifparams = cross_validation_spit(np.concatenate(
    vfats_train["content_smoothed_scaled"].values),
                                 -vfats_train["score"],
                                 IsolationForest(random_state=rng),
                                 param_grid)

# Retrain IF using all unlabelled samples
ifclf = IsolationForest(max_samples=ifparams.max_samples,
                        n_estimators=ifparams.n_estimators,
                        contamination=ifparams.contamination,
                        random_state=rng)

ifclf.fit(np.concatenate(gem_vfats[np.isnan(
    gem_vfats.score)]["content_smoothed_scaled"].values));
    
    


# In[30]:

# Train BDT if you fancy...

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.metrics import classification_report, roc_auc_score

# bdtclf = DecisionTreeClassifier(class_weight="balanced",
#                              random_state=rng)

# abdtclf = AdaBoostClassifier(bdtclf)

# abdtclf.fit(np.concatenate(
#     vfats_train["content_smoothed_scaled"].values),
#             vfats_train["score"])

# print(abdtclf.get_params())

# vfats_test["bdt_score"] = abdtclf.decision_function(
#     np.array(np.concatenate(
#         vfats_test["content_smoothed_scaled"].values)))


# In[31]:

vfats_test["svm_score"] = -svmclf.decision_function(
    np.concatenate(vfats_test["content_smoothed_scaled"].values))
vfats_test["if_score"] = -ifclf.decision_function(
    np.concatenate(vfats_test["content_smoothed_scaled"].values))

get_roc_curve(vfats_test,[("Variance", "variance_score"),                           
                           ("IF", "if_score"),
                           ("SVM", "svm_score"),
                           ("Sobel", "sobel_score")])


# ### Benchmarking neural networks

# In[32]:

# Start Keras session

sess = tf.Session()
K.set_session(sess)


# In[33]:

# Generate CNN input

def score_to_array(score):
    if score == -1:
        return np.asarray([1, 0])
    return np.asarray([0, 1])

def generate_input():  
    return (np.array(np.concatenate(neural_train.content_scaled.values)).reshape(-1, 32),
            np.concatenate(neural_train["score"].apply(score_to_array).values).reshape(-1, 2),
            np.array(np.concatenate(neural_val.content_scaled.values)).reshape(-1, 32),
            np.concatenate(neural_val["score"].apply(score_to_array).values).reshape(-1, 2),
            np.array(np.concatenate(vfats_test.content_scaled.values)).reshape(-1, 32))

(train_x, train_y, val_x, val_y, test_x) = generate_input()


# In[126]:

# Define networks

def shallow_neural_network():
    model = Sequential()
    model.add(Reshape((32, 1), input_shape=(32,), name="input_snn"))
    model.add(Flatten(name="flatten_snn"))
    model.add(Dense(16, name="dense_snn", activation="relu"))
    model.add(Dense(2, activation="softmax", name="output_snn"))
    return model

def convolutional_neural_network():
    model = Sequential()
    model.add(Reshape((32, 1), input_shape=(32,), name="input_cnn"))
    model.add(Conv1D(10, 3, strides=1, padding="valid", name="convolution_cnn", activation="relu"))
    model.add(MaxPooling1D(pool_size=5, strides=5, padding="valid", name="polling_cnn"))
    model.add(Flatten(name="flatten_cnn"))
    model.add(Dense(8, name="dense_cnn", activation="relu"))
    model.add(Dense(2, activation="softmax", name="output_cnn"))
    return model

snn = shallow_neural_network()
cnn = convolutional_neural_network()
print("Shallow Neural Network Architecture:")
snn.summary()
print("Convolutional Network Architecture:")
cnn.summary()


# In[35]:

# Train neural networks

def train_nn(model, x, y, batch_size, loss, name, validation_data=None, 
             validation_split=0.0, class_weight=None):

    model.compile(loss=loss, optimizer="adam")

    early_stopper = EarlyStopping(monitor="val_loss",
                                  patience=32,
                                  verbose=True,
                                  mode="auto")
    
    checkpoint_callback = ModelCheckpoint(("%s/%s.h5" % (models_directory, name)),
                                          monitor="val_loss",
                                          verbose=False,
                                          save_best_only=True,
                                          mode="min")
    return model.fit(x, y,
                     batch_size=batch_size,
                     epochs=8192,
                     verbose=False,
                     class_weight=class_weight,
                     initial_epoch=0,
                     shuffle=True,
                     validation_split=validation_split,
                     validation_data=validation_data,
                     callbacks=[early_stopper, checkpoint_callback])


# In[32]:

# Calculate weights for distierent classes

cw = class_weight.compute_class_weight("balanced",
                                       np.unique(np.argmax(train_y, axis=1)),
                                       np.argmax(train_y, axis=1))
cw = {0: cw[0], 1: cw[1]}


# In[37]:

# Train SNN, this may take some time...


history_snn = train_nn(snn,
                       train_x,
                       train_y,
                       len(train_x),
                       keras.losses.categorical_crossentropy,
                       "snn",
                       validation_data=(val_x, val_y),
                       class_weight=cw)
    
history_snn = history_snn.history
#    
#    with open("%s/history-snn.pkl" % models_directory, "wb+") as history:
#        pickle.dump(history_snn, history)
#
#else:
#    with open("%s/history-snn.pkl" % models_directory, "rb") as history:
#     history_snn = pickle.load(history)


# In[38]:

# Train CNN, this may take some time...

#if not LOAD_MODELS:
history_cnn = train_nn(cnn,
                       train_x,
                       train_y,
                       len(train_x),
                       keras.losses.categorical_crossentropy,
                       "cnn",
                       validation_data=(val_x, val_y),
                       class_weight=cw)

history_cnn = history_cnn.history
    
#    with open("%s/history-cnn.pkl" % models_directory, "wb+") as history:
#        pickle.dump(history_cnn, history)
#
#else:
#    with open("%s/history-cnn.pkl" % models_directory, "rb") as history:
#        history_cnn = pickle.load(history)


# In[39]:

# Plot loss vs. epoch

def plot_loss(data):
    """Plots the training and validation loss"""
    fig, ax = plt.subplots()

    plt.xlabel("Epoch", horizontalalignment='right', x=1.0)
    plt.ylabel("Cross-entropy", horizontalalignment='right', y=1.0)

    plt.plot(data["loss"], linestyle=line_styles[0], color=color_palette["Indigo"][800])
    plt.plot(data["val_loss"], linestyle=line_styles[2], color=color_palette["Orange"][400])
    plt.legend(["Training data set", "Validation data set"], loc="upper right", frameon=False)
    plt.yscale("log")
    
    fig.savefig("%s/local_loss.eps" % figure_export_directory, bbox_inches="tight")
    plt.show();

plot_loss(history_snn)
plot_loss(history_cnn)


# In[40]:

# Export models as .pb, needed for CMSSW implementation:

# outputs = [cnn.output.name.split(":")[0]]
# print("Input name: %s" % cnn.inputs[0].name.split(":")[0])
# print("Output name: %s" % outputs[0])
# constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), outputs);
# tf.train.write_graph(constant_graph, "../model-new", "constantgraph.pb", as_text=False);


# In[41]:

# Reload models from saved

snn_model = load_model("%s/snn.h5" % models_directory)
cnn_model = load_model("%s/cnn.h5" % models_directory)


# In[42]:

# Calculate score for trained networks:

vfats_test["snn_score"] = snn_model.predict(np.array(test_x))[:, 1]
vfats_test["cnn_score"] = cnn_model.predict(np.array(test_x))[:, 1]

m_names = ["sobel","variance","cnn","snn","if","svm"]
ch_distis = [1710, 1711, 1712, 1713, 1714]

for ch_disti in ch_distis:
    for ch_run in range(0,100):
        for mname in m_names:
            for vn in range(1,25):
                te = vfats_test[(vfats_test.disti == ch_disti) & (vfats_test.run == ch_run) & (vfats_test.vfat == vn)]
                if te.empty:
                    continue
                print ch_disti, ch_run, vn, mname, te["%s_score" % mname]

# In[43]:


specificity, sensitivity = benchmark(vfats_test["score"] == 1, vfats_test["cnn_score"] > 0.5)


# In[44]:

get_roc_curve(vfats_test,
              [("CNN", "cnn_score"),
               ("SNN", "snn_score"),
               ("Variance", "variance_score"),
               ("IF", "if_score"),
               ("SVM", "svm_score"),
               ("Sobel", "sobel_score")],
               #("Decision Tree", "bdt_score")],
              (specificity, sensitivity))


# In[45]:

# Score distribution

filter = vfats_test["score"] == 1

fig, ax = plt.subplots()

plt.hist(vfats_test["cnn_score"],
         label = "All samples",
         facecolor = "None",
         histtype = 'step',
         edgecolor = color_palette["Indigo"][800],
         bins= np.arange(0, 1.05, 0.05))

plt.hist(vfats_test["cnn_score"][filter],
         label = "Anomalous samples",
         facecolor = "None",
         histtype = 'step',
         edgecolor = color_palette["Orange"][400],
         bins=np.arange(0, 1.05, 0.05))

plt.legend(frameon=False)
plt.yscale("symlog")
plt.xlabel("Score", horizontalalignment='right', x=1.0)
plt.ylabel("Layers", horizontalalignment='right', y=1.0)
plt.ylim(0, 3000)
plt.subplots_adjust(hspace=0.9)

fig.savefig("%s/local_score.eps" % figure_export_directory)
plt.show();


# In[46]:

# Show convolutional filters

weights = cnn_model.layers[1].get_weights()[0]

for vfat_filter in range(weights.shape[2]):
    plt.imshow(weights[:,:,vfat_filter].reshape(1,-1), interpolation='nearest', vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# ## Checking stability

# In[48]:

# Load data
# Had to be generated before loading in the notebook because ROOT/Notebook import issues

total_no_vfats = 2721
total_no_chambers = 250
runs = [306777, 306793, 306794]

gem_alarms = pd.DataFrame()

for run in runs:
    print("Loading %s" % run),
    path = "%s/%sST.csv" % (data_directory, run)
    gem_alarms = gem_alarms.append(pd.read_csv(path),
                                                   ignore_index=True);
print("Done          "),


# In[50]:

# Add data points for lumi=0

for run in runs:
    df_zero = pd.DataFrame({"lumi": [0],
                            "current_dqm": [total_no_chambers],
                            "total": [0],
                            "emerging": [0],
                            "run": run})
    gem_alarms = pd.concat([df_zero, gem_alarms],
                                   ignore_index=True,
                                   sort=False)


# In[51]:

# Calculate fraction of alarms

gem_alarms["total"] = gem_alarms["total"] / total_no_vfats
gem_alarms["emerging"] = gem_alarms["emerging"] / total_no_vfats
gem_alarms["current_dqm"] = gem_alarms["current_dqm"] / total_no_chambers


# In[54]:

# Plot stability vs. lumi

fig, ax = plt.subplots()

for i, (run, line_type) in enumerate(zip(runs, line_styles)):
    df = gem_alarms[gem_alarms.run == run].sort_values("lumi",
                                                                       ascending=1)

    plt.plot(df["lumi"],
             df["total"],
             linestyle=line_type,
             color=color_palette["Indigo"][900],
             alpha=1,
             label="CNN: Total" if i == 0 else "")

    plt.plot(df["lumi"],
             df["emerging"],
             linestyle=line_type,
             color=color_palette["Indigo"][200],
             alpha=0.6,
             label="CNN: Emerging" if i == 0 else "")

    plt.plot(df["lumi"],
             df["current_dqm"],
             linestyle=line_type,
             color=color_palette["Orange"][700],
             alpha=0.7,
             label="Production" if i == 0 else "")

plt.xlabel("Lumisection", horizontalalignment='right', x=1.0)
plt.ylabel("Fraction of alarms", horizontalalignment='right', y=1.0)

plt.yscale("log")

legend_helper =[]
for i in range(len(runs)):
    legend_helper.append(mlines.Line2D([],
                                       [],
                                       linestyle=line_styles[i],
                                       color="black",
                                       label="Run %s" % runs[i]))

legend_runs = plt.legend(handles=legend_helper, frameon=False, bbox_to_anchor=(0.3, 0.7))

plt.legend(frameon=False, bbox_to_anchor=(0.6, 0.7))
plt.gca().add_artist(legend_runs)
fig.savefig("%s/local_stability.eps" % figure_export_directory, bbox_inches="tight")
plt.show()


# In[ ]:



