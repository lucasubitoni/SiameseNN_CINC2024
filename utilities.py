#Functions For IUGR Healthy classification
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils.layer_utils import get_source_inputs

#generali

def find_n_windows(l_signal,w_window=2400,Maxoverlap=0.5):
    n_windows=np.ceil(l_signal/w_window)
    overlap=np.ceil(((n_windows*w_window)-l_signal)/(n_windows-1))
    if overlap*(n_windows-1)>(w_window*Maxoverlap):
        n_windows=n_windows-1
        overlap=0
    return n_windows,overlap


def WindowSignals(X,y,segnali, Maxoverlap=0.5, Test = False, id_prestazione = False):
    
    X_post_win = []
    y_post_win = []

    if id_prestazione:
        id_prestazione_post_win = []
    
    for i in range(len(X)):
    
        if id_prestazione:
            ID = np.array(X[i][0])
        GestAge = np.array(X[i][1])
        l_signal = len(X[i][2])
        w_window = 2400 
        
        if Maxoverlap>0:
            n_windows,overlap = find_n_windows(l_signal, w_window, Maxoverlap)
        else:
            n_windows = np.floor(l_signal/w_window)
            overlap = 0

        #per fare majority voting serve che siano dispari
        if Test:
            if n_windows%2==0:
                n_windows = n_windows + 1
                overlap=np.ceil(((n_windows*w_window)-l_signal)/(n_windows-1))

        for n in range(int(n_windows)):
            start_index = int(n * (w_window - overlap))
            end_index = int(start_index+w_window)
            to_append = [GestAge] + [X[i][j][start_index:end_index] for j in range(2, 2 + segnali)]
            X_post_win.append(to_append)
            y_post_win.append(y[i])
            if id_prestazione:
                id_prestazione_post_win.append(ID)

    if id_prestazione:
        return np.array(X_post_win, dtype=object), np.array(y_post_win), np.array(id_prestazione_post_win)
    else:
        return np.array(X_post_win,dtype=object), np.array(y_post_win)


def standardizzaPerElemento(X):
    X_stand = []
    i = 0
    invalid = []
    for element in X:
        stand = (element - np.mean(element))/np.std(element)
        X_stand.append(stand)
        if np.isnan(stand).any():
            invalid.append(i)
        i += 1        
            
    return np.array(X_stand), invalid


def StandardizzaERiorganizzaTrain(X_train, y_train, segnali, scaler):
    
    #questa parte è da mettere a posto

    #GestAge
    GestAge_train = X_train[:, 0].reshape(-1, 1)
    GestAge_train_scaled = scaler.fit_transform(GestAge_train)
    train_data_GestAge = GestAge_train_scaled 

    #FHR    
    X_train_FHR=X_train[:,1+segnali.index('FHR120bpm')] 
    train_data_FHR = np.array(X_train_FHR.tolist())
    scaling_factor_mean_FHR=np.mean(train_data_FHR)
    scaling_factor_std_FHR= np.mean(np.std(train_data_FHR, axis=1))
    train_data_stand_FHR=(train_data_FHR-scaling_factor_mean_FHR)/scaling_factor_std_FHR

    dfScaledTrain = pd.DataFrame({
        'GestAge': train_data_GestAge.tolist(),
        'FHR': train_data_stand_FHR.tolist(),
    })

    #TOCO
    if "TOCO" in segnali:
        X_train_TOCO=X_train[:,1+segnali.index('TOCO')] 
        train_data_stand_TOCO, invalidTrain =  standardizzaPerElemento(X_train_TOCO)
        dfScaledTrain['TOCO'] = train_data_stand_TOCO.tolist()
    
    #FMP
    if "FMP" in segnali:
        train_data_FMP=X_train[:,1+segnali.index('FMP')] 
        dfScaledTrain['FMP'] = train_data_FMP.tolist()    
    
    #Label
    dfScaledTrain['label'] = y_train    

    if "TOCO" in segnali:
        dfScaledTrain = dfScaledTrain.drop(invalidTrain)

    scalingfactorsFHR = [scaling_factor_mean_FHR, scaling_factor_std_FHR]

    return dfScaledTrain, scalingfactorsFHR, scaler


def StandardizzaERiorganizzaTest(X_test, y_test,segnali, scaler, scalingfactorsFHR):

    #GestAge
    GestAge_test = X_test[:, 0].reshape(-1, 1)
    GestAge_test_scaled = scaler.transform(GestAge_test)
    test_data_GestAge = GestAge_test_scaled 

    #FHR
    X_test_FHR=X_test[:,1+segnali.index('FHR120bpm')]
    test_data_FHR = np.array(X_test_FHR.tolist())
    test_data_stand_FHR=(test_data_FHR-scalingfactorsFHR[0])/scalingfactorsFHR[1]

    dfScaledTest = pd.DataFrame({
        'GestAge': test_data_GestAge.tolist(),
        'FHR': test_data_stand_FHR.tolist(),
    })

    #TOCO
    if "TOCO" in segnali:
        X_test_TOCO=X_test[:,1+segnali.index('TOCO')]
        test_data_stand_TOCO, invalidTest = standardizzaPerElemento(X_test_TOCO)
        dfScaledTest['TOCO'] = test_data_stand_TOCO.tolist()
    
    #FMP
    if "FMP" in segnali:
        test_data_FMP=X_test[:,1+segnali.index('FMP')]
        dfScaledTest['FMP'] = test_data_FMP.tolist()

    dfScaledTest['label'] = y_test

    if "TOCO" in segnali:
        dfScaledTest = dfScaledTest.drop(invalidTest)

    return dfScaledTest





