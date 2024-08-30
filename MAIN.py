# ========================================================================================================
#                      RUN WITHOUT HYPERPARAMETERS SEARCH 
# ========================================================================================================
#   Giulio Steyde, Luca Subitoni, 04/04/2024 

# ---- imports ---------------------------

import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from Classes import *
from utilities import *
from siamese_utilities import *
from sklearn.neighbors import KNeighborsClassifier
import itertools
from sklearn.metrics import balanced_accuracy_score
from warnings import simplefilter
import pickle
import tensorflow
from tensorflow.keras.optimizers import Adadelta, Adam
import ast
import joblib
import logging
import sys
from sklearn.metrics import roc_auc_score

# ---- settings ---------------------------

directory_name = os.getcwd()  #Set directory to where code and data are located. ADJUST FOR YOUR MACHINE
gpu_number = 1 #Set the GPU number

# ---- allocate GPU
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[gpu_number], 'GPU')
logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")


# -----------------------------------------
# set current directory to indicated folder
os.chdir(directory_name)

# ----------------------   load Data
df = pd.read_pickle('DATA')

#  ---------------------   a-priori hyperparameters
n_neighbors = 1
segnali = ['FHR120bpm','FMP'] 
maxBatch =  120 #max depends on hardware .
test_size = 0.2 
overlap = 0.33

# ----------------------   Ignore Warnings
simplefilter(action='ignore', category=FutureWarning)


# ========================================================================================================
# DATASET
# ========================================================================================================

# ----------------------   Dataset   -------------------------
df = df.sample(frac=1, random_state=42, ignore_index=True)
df.columns = ['patnum','id_prestazione','sett_gestazione','label','FHR120bpm','TOCO','FMP']

# ----------------------   Preprocessing   -------------------------
to_convert = ['patnum', 'label', 'id_prestazione','sett_gestazione']
df[to_convert] = df[to_convert].astype(int)

# train/validation/hold-out test test
colonne = ['id_prestazione','sett_gestazione'] + segnali
X = df[colonne].values 
y = df['label'].values
groups = df['patnum'].values

# Two recordings of the same patient cannot be in two different subsets
gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

# Define TrainAll /Test
for train_idx, test_idx in gss.split(X, y, groups=groups):
    X_trainAll, X_test = X[train_idx], X[test_idx]
    y_trainAll, y_test = y[train_idx], y[test_idx]
    groups_trainAll, groups_test = groups[train_idx], groups[test_idx]

# Divide TrainAll in Train and Validation
gss_train = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, val_idx in gss_train.split(X_trainAll, y_trainAll, groups=groups_trainAll):
    X_train, X_val = X_trainAll[train_idx], X_trainAll[val_idx]
    y_train, y_val = y_trainAll[train_idx], y_trainAll[val_idx]

# Window signals in 2400 points
X_train,   y_train        = WindowSignals(X_train, y_train, len(segnali), Maxoverlap=overlap)
X_val,     y_val          = WindowSignals(X_val,   y_val  , len(segnali), Maxoverlap=overlap)
X_trainAll,y_trainAll     = WindowSignals(X_trainAll,y_trainAll, len(segnali), Maxoverlap=overlap)
X_test,    y_test, idTest = WindowSignals(X_test, y_test,len(segnali), Maxoverlap=overlap, Test = True, id_prestazione = True)

# Standardization -------------------------------------------
#TRAIN
scaler = StandardScaler()
dfScaledTrain, scalingfactorsFHR, scaler = StandardizzaERiorganizzaTrain(X_train, y_train, segnali, scaler)
#VALIDATION
dfScaledVal = StandardizzaERiorganizzaTest(X_val, y_val,segnali, scaler, scalingfactorsFHR)
#TRAIN ALL
scaler2 = StandardScaler()
dfScaledTrainAll, scalingfactorsFHRAll, scaler2 = StandardizzaERiorganizzaTrain(X_trainAll, y_trainAll, segnali, scaler2)
#TEST
dfScaledTest = StandardizzaERiorganizzaTest(X_test, y_test,segnali, scaler2, scalingfactorsFHRAll)


# ========================================================================================================
#   TRAINING
# ========================================================================================================

# define hyperparameters

kernels_size =   [8,5,3]
batch_size   =   35
epochs =         500
learning_rate =  1e-5
n_feature_maps = 38  
dropout =        0.6
output_size =    10
architecture =   'Resnet_GA_3'
noise =          0
activation =     'leaky_relu'

optimizer = Adam(learning_rate=learning_rate)

#create network object
FinalNet = SiameseNetwork(reinizializza = 1, Architecture = architecture, n_feature_maps = n_feature_maps, activation = activation, kernels_size = kernels_size, optimizer = optimizer, dropout = dropout, output_size = output_size, input_shape = (2400,len(segnali)), n_neighbors=n_neighbors)
FinalNet.maxBatch = maxBatch

#train
FinalNet.fit(dfScaledTrain, dfScaledVal, scaler, epochs, batch_size, reset_best_weights=True, Measure_time=False, LimitComb = 32)


# ========================================================================================================
#   TESTING
# ========================================================================================================

#Plot projection
FinalNet.PlotProjection(dfScaledTrainAll, 'PCA', dfScaledTest, type = 'PCA')



# See Accuracy and AUC
PR_TrainAll = FinalNet.ProjectData(dfScaledTrainAll)
PR_Test     = FinalNet.ProjectData(dfScaledTest)

BalAcc = []
AUC    = []

for i in range(1,21):

    KNNi = KNeighborsClassifier(n_neighbors = i, metric='euclidean') 
    KNNi.fit(PR_TrainAll, np.array(dfScaledTrainAll['label'].values).astype('int'))

    y_pred = KNNi.predict(PR_Test)
    y_pred_proba = KNNi.predict_proba(PR_Test)
    y_pred_proba = y_pred_proba[:,1]

    y_test = np.array(dfScaledTest['label'].values).astype('int')
    
    TestBalAccuracy = balanced_accuracy_score(y_test, y_pred) 
    BalAcc.append(TestBalAccuracy)
    print(TestBalAccuracy)

    temp = roc_auc_score(y_test, y_pred_proba)
    print(temp)
    AUC.append(temp)


x = range(1,21)
plt.figure(figsize=(6, 3))

plt.subplot(1,2,1)
plt.plot(x, BalAcc, marker='o', linestyle='-', color='k', label='Test Accuracy', markersize = 4, lw=0.5)
plt.ylim([0, 1])
plt.xticks(ticks=range(1,21,2))
y_ticks = np.arange(0, 1.1, 0.1)
plt.yticks(ticks=y_ticks)
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.xlabel('Number of neighbours', fontsize=12)
plt.tight_layout(pad=1.5)
plt.title('Accuracy', fontsize = 12)

plt.subplot(1,2,2)
plt.plot(x, AUC,  marker='o', linestyle='-', color='k', label='Test AUC', markersize = 4, lw=0.5)
plt.ylim([0, 1])
plt.xticks(ticks=range(1,21,2))
plt.yticks(ticks=y_ticks)
plt.tight_layout(pad=1.5)
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.xlabel('Number of neighbours', fontsize=12)
plt.title('AUC', fontsize=12)



