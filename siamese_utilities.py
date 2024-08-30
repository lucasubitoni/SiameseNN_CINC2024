import numpy as np
import pandas as pd
import Classes
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from itertools import permutations
import random
from collections import Counter
import time
from sklearn.metrics import balanced_accuracy_score
import os
import csv
#from numba import jit

# ----------------------------------------------------------------------------------------------------------------
#                               Functions Called Directly by MainSiamese                                         -
# ----------------------------------------------------------------------------------------------------------------

def KNNAccuracy(dfScaledTrain, dfScaledTest, model, n_neighbors, Test = False, idTest = None):

    '''
    Train and Test a simple KNN classifier that takes as input the projection of the Siamese Network
    '''
    
    #project
    PR_TrainAll = model.ProjectData(dfScaledTrain)
    PR_Test     = model.ProjectData(dfScaledTest)
    KNN2 = KNeighborsClassifier(n_neighbors = n_neighbors, metric='euclidean') 
    KNN2.fit(PR_TrainAll, np.array(dfScaledTrain['label'].values).astype('int'))

    y_pred = KNN2.predict(PR_Test)
    y_test = np.array(dfScaledTest['label'].values).astype('int')
    TestBalAccuracy = balanced_accuracy_score(y_test, y_pred) 

    print("Val Bal Accuracy: " + str(TestBalAccuracy))

    if Test:

        #majority voting on all available segments of the same recording
        y_pred_majority = []
        y_true_majority = []

        unique_ids = set(idTest)
        for unique_id in unique_ids:
            indices = [i for i, id_value in enumerate(idTest) if id_value == unique_id]
            majority_pred = Counter(y_pred[indices]).most_common(1)[0][0]
            majority_true = Counter(y_test[indices]).most_common(1)[0][0]
            y_pred_majority.append(majority_pred)
            y_true_majority.append(majority_true)

        # Calculate accuracy
        TestBalAccuracy_majority = balanced_accuracy_score(y_true_majority, y_pred_majority)
        print("Majority Voting Test Bal Accuracy: " + str(TestBalAccuracy_majority))

        return TestBalAccuracy_majority
    else:
        return TestBalAccuracy


def extract_first_element(row, name):
    return row.name, row[name][0]



# This function is fast but consumes A LOT of memory. Do not run on laptop (maybe on server? TODO: test)

def generate_triplets(df, limite_giorni):

    indices = np.array(df['Index'], dtype=np.int16)

    classes = np.array(df['label'])
    
    df['GA'] = df['GA'].astype(np.int16)

    # Generate all permutations of indices (order matters) esploderà la memoria? testare -> sì esplode
    combos = np.array(list(permutations(indices, 3)))

    # Filter combinations based on the condition1 -> ++- o --+ (prende queste)
    mask1 = (classes[combos[:, 0]] == classes[combos[:, 1]]) & (classes[combos[:, 0]] != classes[combos[:, 2]])

    combos = combos[mask1]

    # Filter combinations based on the condition2 -> similar GA (prende queste)
    mask2 = (np.abs(df['GA'][combos[:, 0]] - df['GA'][combos[:, 1]]) < limite_giorni) and (np.abs(df['GA'][combos[:, 0]] - df['GA'][combos[:, 2]]) < limite_giorni)

    triplets = combos[mask2]

    return triplets

