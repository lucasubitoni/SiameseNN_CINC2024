# Import libraries
import numpy as np
import random
import tensorflow as tf
import keras
from tensorflow.keras.models import  load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
import time
from numpy import dot
from numpy.linalg import norm
from sklearn.neighbors import KNeighborsClassifier
import os
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import architectures
from siamese_utilities import *  
from sklearn.utils import shuffle        


# ==========================================================================
# =========================== SIAMESE NETWORK CLASS ========================
# ==========================================================================

class SiameseNetwork:

    """
    Class to create the siamese neural network
    """

    # Initialization of the Siamese Network
    def __init__(self,reinizializza = 1, Architecture = 'Resnet_GA_3', n_feature_maps = 64, activation = 'leaky_relu', kernels_size = [8,5,3], optimizer = SGD(learning_rate=1e-3), dropout = 0.2, output_size = 10, input_shape = (2400,2), n_neighbors=1):

        """
        This function is used when the network object is called in order to initialize all of its attributes
        """

        # Boolean Variable (1=True; 0=False) to tell if I need to reinitialize the network
        if reinizializza:

            # Input size of the network
            input_shape = input_shape 
            input_shapeGA = (1)

            # Choosing the architecture of the network
            if Architecture == 'Resnet_GA_3':
                modello = architectures.Resnet_GA_3(input_shape,input_shapeGA,n_feature_maps, activation, kernels_size, optimizer, dropout, output_size)

            elif Architecture == 'Resnet_GA_2':
                modello = architectures.Resnet_GA_2(input_shape,input_shapeGA,n_feature_maps, activation, kernels_size, optimizer, dropout, output_size)

            elif Architecture == 'Resnet_GA_1':
                modello = architectures.Resnet_GA_1(input_shape,input_shapeGA,n_feature_maps, activation, kernels_size, optimizer, dropout, output_size)    
            
            elif Architecture == 'Resnet_3':
                modello = architectures.Resnet_3(input_shape,n_feature_maps,activation,kernels_size, optimizer, dropout, output_size)
            
            elif Architecture == 'Vanilla':
                modello = architectures.Vanilla(input_shape,n_feature_maps,activation,kernels_size, optimizer, dropout, output_size) 

            elif Architecture == 'Resnet_GA_4':
                modello = architectures.Resnet_GA_4(input_shape,input_shapeGA,n_feature_maps, activation, kernels_size, optimizer, dropout, output_size)       
            
            elif Architecture == 'Resnet_GA_3s':
                modello = architectures.Resnet_GA_3s(input_shape,n_feature_maps, activation, kernels_size, optimizer, dropout, output_size)       
            else:
                print(' Sorry this is work in progress :( ')
                #non yet existing architecture
                return
        else:
            print(' Sorry this is work in progress :( ')
            #import existing model to partially retrain it
            return
    
        # Print the model summary
        modello.summary()

        # Setting the network object attributes
        self.model = modello
        self.GA = True if Architecture == 'Resnet_GA_4' or Architecture == 'Resnet_GA_3' or Architecture == 'Resnet_GA_2' or Architecture == 'Resnet_GA_1' else False
        self.margin = 0.2
        self.n_neighbors = n_neighbors

# ==============================================================================================================================================  

    # Training the Siamese Network
    def fit(self,DatiTrain_df, DatiValidation_df, scaler, epoche,batch_size, reset_best_weights = True, save_model = False, plot_loss = False, noise = 0, Measure_time = False, LimitComb = np.inf):

        """
        This function is used to fit the initialized neural network architecture
        """

        #Flag: diventa 0 se qualcosa non va
        self.stato = 1 

        # Set the validation data
        DatiValidation = SiameseDataset(DatiValidation_df,scaler)
        
        #self.PlotProjection(DatiTrain_df, 'prova', DatiValidation_df, type = 'PCA') #plot for debugging

        # Define metrics to be tracked
        current_lossTrain = 0                       
        self.EpochLossTrain = np.array([]) 
        self.BestEpoch = 0
        #self.EasyTriplets = np.array([])
        #EasyTriplets = np.array([])

        self.ClassDistanceVal = np.array([])
        self.ClassDistanceTrain = np.array([])
        self.HighestDistance = -np.inf
 
        soglia = -self.margin

        # Cycling on the numbers of epochs
        for epoch in range(epoche):

            T = np.zeros(4)

            # Shuffling train dataset on each epoch
            DatiTrain_df = shuffle(DatiTrain_df)

            # Computing the number of batches that I have available given the selected batch size and the size of the training dataset
            num_batches = len(DatiTrain_df) // batch_size

            # Cycling on the various batches 
            for batch_index in range(num_batches):

                # If I want to measure time I keep track of it
                if Measure_time:
                    start_batchTime = time.time()

                # Computing the start and end index of the chosen batch
                start_index = batch_index * batch_size
                end_index = (batch_index + 1) * batch_size

                # Retrieving batch data given my start and end indexes I computed above
                BatchData = DatiTrain_df.iloc[start_index:end_index]

                # Create the dataset from the selected batch
                DatiTrain = SiameseDataset(BatchData,scaler)         

                # Generate the various combinations for training the Siamese Network (lista di tutte le possibili combinazioni di triplette all'interno del batch che ho scelto --> Quello più grosso)
                BatchDataTuplesAll = DatiTrain.GenerateCombList(100, Balance = True)

                # Inside the batch I made the combinations, now INSIDE the combinations I extract OTHER BATCHES
                if len(BatchDataTuplesAll) > self.maxBatch:

                    # Here I process the combinations within the dataset using minibatches.
                    # I call the network in inference mode (sig_1, sig_2, sig_3 are the signals formatted as expected by the Siamese network).
                    # I iterate over the batch in blocks of size max_batch (the largest size that fits in memory), one block at a time.
                    # sig_1, sig_2, sig_3 (conceptually A, P, N) are tensors, each containing multiple signals.
                    # Only distances above a certain threshold (i.e., the hard examples for the network) are considered valid.
                    # Valid signals are appended to the BatchDataTuples list.
                    # To avoid discarding data, a final check on the remaining signals (which may be fewer than max_batch) is also performed.

                    esci = 0

                    # create the tensor which will be sliced later on
                    DatiTrain.createTensor(self.GA) 
  
                    #triplet mining: compute the distances and keep only those > 0
                    start_timeMining = time.time()
                    ministeps = int(np.floor(np.min([LimitComb,len(BatchDataTuplesAll)//self.maxBatch])))
                    BatchDataTuples = []
                    
                    for i in range(ministeps):
           
                        sig_1, sig_2, sig_3 = DatiTrain.createTensorComb(BatchDataTuplesAll[i*self.maxBatch:(i+1)*self.maxBatch], self.GA) 
                        ap_distance, an_distance = self.model([sig_1, sig_2, sig_3], training = False)
                        valid =  ap_distance - an_distance > soglia
                        BatchDataTuples.extend([BatchDataTuplesAll[i*self.maxBatch+j] for j in range(self.maxBatch) if valid[j]])

                    # Process remaining combinations if total number is not divisible by batch size 
                    if not(LimitComb<np.inf):
                        if len(BatchDataTuplesAll) % self.maxBatch != 0:   
                       
                            sig_1, sig_2, sig_3 = DatiTrain.createTensorComb(BatchDataTuplesAll[ministeps*self.maxBatch:], self.GA)
                            ap_distance, an_distance = self.model([sig_1, sig_2, sig_3], training = False)
                            valid =  ap_distance - an_distance > soglia
                            BatchDataTuples.extend([BatchDataTuplesAll[ministeps*self.maxBatch+j] for j in range(len(BatchDataTuplesAll) % self.maxBatch) if valid[j]])

                    if Measure_time:
                        TimeMining = time.time() - start_timeMining
                    
                    # Vettore che si aggiorna ogni tot e salvo la proporzione di triplette facili
                    #EasyTriplets = np.append(EasyTriplets, 1-len(BatchDataTuples)/len(BatchDataTuplesAll))

                    if Measure_time:
                        start_timeTraining = time.time()

                    # Handle training on minibatches to avoid memory overflow
                    # Steps tells us how many "loops" to do in the DataTuples list
                    steps = len(BatchDataTuples)//self.maxBatch

                    # If no hard triplets are found, set loss to zero
                    if steps == 0:
                        loss_step = tf.cast(0.0, tf.float32) # All correct
                    else:
                        loss_step = np.empty(steps)
                        for step in range(steps):
                            # Among difficult signals (i.e. the valid ones), I take a mini-batch of maxBatch size to train the net
                            start_index = step * self.maxBatch
                            end_index = (step + 1) * self.maxBatch
                            BatchDataTuplesStep = BatchDataTuples[start_index:end_index]
                            if noise > 0:
                                sig_1, sig_2, sig_3 = DatiTrain.getSiamaseDatasetNoise(BatchDataTuplesStep, self.GA, noise = noise)
                            else:
                                sig_1, sig_2, sig_3 = DatiTrain.createTensorComb(BatchDataTuplesStep, self.GA)

                            loss_step[step] = self.train_step(sig_1, sig_2, sig_3).numpy()
                            
                        loss_step = np.mean(loss_step)
                    if np.isnan(loss_step):
                        print('NAN LOSS!!!')
                        esci = 1
                    else:    
                        # The mean is iteratively updated
                        current_lossTrain = current_lossTrain * (batch_index/(batch_index+1)) + (loss_step/(batch_index+1))
                else:
                    esci = 1
                
                # If I wanted to keep track of time, now I take a snapshot and print it
                if Measure_time and esci==0:
                    TimeTraining = time.time() - start_timeTraining
                    BatchTime = time.time() - start_batchTime
                    if BatchTime>0:
                        print("Step Time: {} s, Mining: {} %, Training: {} %".format(BatchTime, 100*(TimeMining/BatchTime), 100*(TimeTraining/BatchTime)))
                            
            if Measure_time:
                print("T0: {}, T1: {}, T2: {}".format(T[0],T[1],T[2]))

            # END EPOCH  ------------------------------------------------------- 
            Measure_time = False #solo per la prima epoca

            if esci:
                self.stato = 0
                print('There is a problem!')
                return
                
            # Train loss
            self.EpochLossTrain = np.append(self.EpochLossTrain, current_lossTrain)     
            
            # Distance difference in train
            TrainDistance = self.AvgClassDistance(DatiTrain)
            self.ClassDistanceTrain = np.append(self.ClassDistanceTrain, TrainDistance)

            # Distance difference in validation
            ValDistance = self.AvgClassDistance(DatiValidation)
            self.ClassDistanceVal = np.append(self.ClassDistanceVal, ValDistance)


            if (np.isnan(self.EpochLossTrain).any()) or (np.isnan(TrainDistance).any()) or (np.isnan(ValDistance).any()):
                # Network is not networking
                self.stato = 0
                print('There is a problem!!')
                return

            # Percentage easy triplets        
            #self.EasyTriplets = np.append(self.EasyTriplets, np.mean(EasyTriplets))
            
            #accuracy
            #self.Accuracy = np.append(self.Accuracy,KNNAccuracy(DatiTrain_df, DatiValidation_df, self, 3, Test=False))

            # If I find a model whose highest distance is bigger than the previous best one, than that is my new best model
            if ValDistance>self.HighestDistance:
                self.HighestDistance = ValDistance # Updating the highest distance
                if reset_best_weights: # If I want to save the best model and reset it to the best one when I finish training
                    self.model.save_weights("BestWeightsV.h5")
                    self.BestEpoch = epoch 
         
            # Printing status at the end of the epoch
            print("End of epoch {} Loss {} Class Distance {}".format(epoch + 1 , current_lossTrain, ValDistance))

            # I gradually change the margin while training to make the problem harder
            if current_lossTrain < 0.1:
                soglia = 0
            elif current_lossTrain < 0.15:
                soglia = -self.margin/4
            elif current_lossTrain < 0.17:
                soglia = -self.margin/2
                 
            # Reset metrics
            current_lossTrain = 0 

        # Evaluating the accuracy of the KNN with a fixed N=3 before resetting the model to its best weights
        print('Before Reset')
        self.ValAccuracy = KNNAccuracy(DatiTrain_df, DatiValidation_df, self, self.n_neighbors, Test=False)

        # If I want to save the best model and reset it to the best one when I finish training 
        if reset_best_weights:
            # Load best weights
            self.model.load_weights("BestWeightsV.h5")
            if save_model:
                try:
                    self.model.save("ModelloSiamese.h5")
                except:
                    print("Could not save model!")

            # After resetting the model I want to compute its accuracy to see if everything worked fine
            print('After Reset')
            self.ValAccuracy = KNNAccuracy(DatiTrain_df, DatiValidation_df, self, self.n_neighbors, Test=False)
            
        # If I want to plot the loss curve for the train and validation datasets
        if plot_loss:
            plt.figure()
            plt.plot(self.MonitorLossTrain)
            plt.plot(self.MonitorLossValidation)
            plt.title('Loss')
        return  
    


# ==============================================================================================================================================  
    
    # Computing the validation loss
    
    def GetLoss(self,DatiValidation, CombListValidation, batch_size):

        """
        This function is used to compute the validation loss of the network
        """
       
        # Initializing the validation loss
        ValLoss = 0

        # Computing how many batches I can do from the various combinations obtained from my validation dataset
        num_batchesV = len(CombListValidation)//batch_size
       
        # Cycling on the number of possible batches
        for batch_index in range(num_batchesV):

            # Computing the start, end and retrieving the tuples from the combination array
            start_index = batch_index * batch_size
            end_index = (batch_index + 1) * batch_size
            batch_tuples = CombListValidation[start_index:end_index]

            # Obtain the anchor, positive and negative examples 
            sig_1, sig_2, sig_3 = DatiValidation.getSiamaseDataset(batch_tuples, self.GA)

            # Computing the loss in an iterative way (penso sia una media calcolata in modo iterativo)
            loss_step = self._compute_loss([sig_1, sig_2, sig_3], train = False).numpy()
            ValLoss = ValLoss * (batch_index/(batch_index+1)) + (loss_step/(batch_index+1))

        # Returing the final validation loss
        return ValLoss



# ==============================================================================================================================================  
        
    # Computing the average class distance
    def AvgClassDistance(self, DatiValidation_obj):

        """
        This function computes the average class distance
        """

        # Project validation data (based on row, not on index)
        PR = self.ProjectData(DatiValidation_obj.Data_frame_orig) # --> numpy array
 
        GenerateCoupleListDiff, GenerateCoupleListSame = DatiValidation_obj.GenerateCoupleList(reset_index = True)

        # Calcolo le distanze
        distSame = np.array([self.distance([PR[i[0]]], [PR[i[1]]]) for i in GenerateCoupleListSame])
        distDiff = np.array([self.distance([PR[i[0]]], [PR[i[1]]]) for i in GenerateCoupleListDiff])

        # Chance: 0.6931471805599453. Higher is better
        return np.log1p(np.mean(distDiff)/np.max([np.mean(distSame), 1e-6]))


# ==============================================================================================================================================  

    
    def ProjectData(self,dfScaled):
       
        EmbdeddingModel = self.model.get_layer('EmbeddingModel')
        PR = []
        L = dfScaled.shape[1]

        if self.GA:

            for i in range(len(dfScaled)):
                sig1 = [tf.constant(np.array([dfScaled.iloc[i,j] for j in range(1, (L-1))],dtype=np.float32).T.reshape(2400, L-2), dtype=tf.float32)]
                sig2 = [dfScaled.iloc[i]["GestAge"]]
                sig1 = tf.stack(sig1)
                sig2 = tf.stack(sig2)
                sig = [sig1, sig2]
                pr = EmbdeddingModel(sig).numpy()[0] 
                PR.append(pr.tolist())
        else:

            for i in range(len(dfScaled)):

                sig1 = [tf.constant(np.array([dfScaled.iloc[i,j] for j in range(0,(L-1))], dtype=np.float32).T.reshape(2400, L-1), dtype=tf.float32)]
                sig1 = tf.stack(sig1)
                
                pr = EmbdeddingModel(sig1).numpy()[0] 
                PR.append(pr.tolist())

        return np.array(PR)
    
# ==============================================================================================================================================  
    
    def PlotProjection(self, Data, title, *TestData, type = 'TSNE'):

        """
        Function to project the data to visualize them. Type can be PCA (linear) or TSNE (non-linear)
        """

        labels = np.array(Data['label'].values).astype('int')
        Data = self.ProjectData(Data)
        if type == 'TSNE':
            from sklearn.manifold import TSNE
            TSNE_obj = TSNE(n_components = 2, learning_rate = 'auto', init = 'random', perplexity = 10)
            Data_project = TSNE_obj.fit_transform(Data)
            plt.figure()
            plt.scatter(Data_project[:,0],Data_project[:,1], c = labels)
            plt.title('TSNE ' + title)
        elif type == 'PCA':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            Data_project = pca.fit_transform(Data)
            if TestData:
                label_test = np.array(TestData[0]['label'].values).astype('int')
                TestData_project = pca.transform(self.ProjectData(TestData[0]))
                plt.figure()
                plt.scatter(Data_project[:,0],Data_project[:,1], c = labels)
                plt.title('PCA train' + title)
                plt.figure()
                plt.scatter(TestData_project[:,0],TestData_project[:,1], c = label_test)
                plt.title('PCA test' + title)
            else:
                plt.figure()
                plt.scatter(Data_project[:,0],Data_project[:,1], c = labels)
                plt.title('PCA ' + title)

# ==============================================================================================================================================  
    
    # For each standalone signal 
    #Returns the embedding from "EmbeddingModel" for a single signal (one of the three siamese nets)
    def project(self, segnale):
        EmbdeddingModel = self.model.get_layer('model')
        return EmbdeddingModel(segnale)
    
# ==============================================================================================================================================  

    def distance(self, sig1, sig2):

        """
        This function computes the euclidean distance between the embeddings of the signals
        """

        # Compute euclidean distance
        dist = np.linalg.norm(sig1[0]-sig2[0])

        # Compute cosine similarity distance
        #cos_sim = dot(PR1, PR2)/(norm(PR1)*norm(PR2))
        #dist = 1 - cos_sim

        return dist
    
# ==============================================================================================================================================  

    @tf.function
    def _compute_loss(self, data, train = True):

        """
        Function to compute the triplet loss
        """

        # The output of the network is a tuple containing the distances between the anchor and the positive example, and the anchor and the negative example.
        ap_distance, an_distance = self.model(data, training = train)
        
        # Computing the Triplet Loss by subtracting both distances and making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0) # Remove negative losses (i.e. the easy triplets)
    
        # Get final mean triplet loss over the positive valid triplets
        if tf.size(loss)==0:
            loss = tf.cast(0.0, tf.float32)
        else:
            loss = tf.reduce_mean(loss) 

        # Ritorno la loss calcolata
        return loss 
    
# ==============================================================================================================================================  
    
    @tf.function
    def train_step(self, sig_1, sig_2, sig_3):

        """
        Function to do a train step of the network: loss computation and gradient propagation
        """
        
        with tf.GradientTape() as tape:
            loss = self._compute_loss([sig_1, sig_2, sig_3])     
        
        gradients = tape.gradient(loss, self.model.trainable_variables)     
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    














# ==========================================================================
# =========================== SIAMESE DATASET CLASS ========================
# ==========================================================================

class SiameseDataset():

    """
    Classe che serve per creare il dataset formattato adeguatamente per la rete siamese
    """



    # Quando creo l'oggetto viene chiamato questo metodo quindi questi valori sono quelli che prende in input
    def __init__(self,Data_frame_mio, scaler):

        """
        This function is used when the network object is called in order to initialize all of its attributes
        """

        # I keep in memory the original dataframe
        self.Data_frame_orig = Data_frame_mio
        
        # I keep in memory the scaler
        self.scaler = scaler



    # ============================================================================================================================================== 

    def getSiamaseDataset(self, TupleList, GA):

        """
        Function to get the anchor, positive and negative signals
        """

        # Length of the original dataframe
        L = self.Data_frame_orig.shape[1]

        # If I want to consider the gestational age as an input to the neural network
        if GA:
          
            sig1_1 = [tf.constant(np.array([self.Data_frame_orig.loc[combinazione[0]].iloc[[j]].values.tolist()[0] for j in range(1, (L-1))],dtype=np.float32).T.reshape(2400, L-2), dtype=tf.float32) for combinazione in TupleList]
            sig1_2 = [self.Data_frame_orig.loc[combinazione[0],"GestAge"] for combinazione in TupleList]
            sig1_1 = tf.stack(sig1_1)
            sig1_2 = tf.stack(sig1_2)
            sig1 = [sig1_1, sig1_2]

            sig2_1 = [tf.constant(np.array([self.Data_frame_orig.loc[combinazione[1]].iloc[[j]].values.tolist()[0] for j in range(1,(L-1))], dtype=np.float32).T.reshape(2400, L-2), dtype=tf.float32) for combinazione in TupleList]
            sig2_2 = [self.Data_frame_orig.loc[combinazione[1],"GestAge"] for combinazione in TupleList]
            sig2_1 = tf.stack(sig2_1)
            sig2_2 = tf.stack(sig2_2)
            sig2 = [sig2_1, sig2_2]

            sig3_1 = [tf.constant(np.array([self.Data_frame_orig.loc[combinazione[2]].iloc[[j]].values.tolist()[0] for j in range(1,(L-1))], dtype=np.float32).T.reshape(2400, L-2), dtype=tf.float32) for combinazione in TupleList]
            sig3_2 = [self.Data_frame_orig.loc[combinazione[2],"GestAge"] for combinazione in TupleList]
            sig3_1 = tf.stack(sig3_1)
            sig3_2 = tf.stack(sig3_2)
            sig3 = [sig3_1, sig3_2]

        # If I do not want to consider the gestational age as an input to the neural network
        else:

            sig1 = [tf.constant(np.array([self.Data_frame_orig.loc[combinazione[0]].iloc[[j]].values.tolist()[0] for j in range(1,(L-1))], dtype=np.float32).T.reshape(2400, L-2), dtype=tf.float32) for combinazione in TupleList]
            sig2 = [tf.constant(np.array([self.Data_frame_orig.loc[combinazione[1]].iloc[[j]].values.tolist()[0] for j in range(1,(L-1))], dtype=np.float32).T.reshape(2400, L-2), dtype=tf.float32) for combinazione in TupleList]
            sig3 = [tf.constant(np.array([self.Data_frame_orig.loc[combinazione[2]].iloc[[j]].values.tolist()[0] for j in range(1,(L-1))], dtype=np.float32).T.reshape(2400, L-2), dtype=tf.float32) for combinazione in TupleList]

            sig1 = tf.stack(sig1)
            sig2 = tf.stack(sig2)
            sig3 = tf.stack(sig3)

        # Returning the anchor, positive and negative (o ordine diverso)
        return sig1, sig2, sig3
    

    # ==============================================================================================================================================    
    #27/03/2024 - Efficiency memory modification (GIULIO)
    def createTensor(self, GA):

        #create a tensor from the elements of the dataframe. The input to the net will be the combination of three of its slices (APN) on the basis of the combination list
        #call on the mini-batch of the dataset, not all of it, or it will likely do not fit in memory
        
        # number of signals
        L = self.Data_frame_orig.shape[1]
        
        if GA:

            tensor_A = [tf.constant(np.array([self.Data_frame_orig.loc[id].iloc[[j]].values.tolist()[0] for j in range(1, (L-1))],dtype=np.float32).T.reshape(2400, L-2), dtype=tf.float32) for id in self.Data_frame_orig.index]            
            tensor_B = [self.Data_frame_orig.loc[id,"GestAge"] for id in self.Data_frame_orig.index]
            tensor_A = tf.stack(tensor_A) #non so se serve tbh
            tensor_B = tf.stack(tensor_B)
            tensorAll = [tensor_A, tensor_B]

        else:
            #tensor_A = [tf.constant(np.array([self.Data_frame_orig.iloc[id].iloc[[j]].values.tolist()[0] for j in range(1, L-1)],dtype=np.float32).T.reshape(2400, L-2), dtype=tf.float32) for id in range(len(self.Data_frame_orig))]            
            #così usa GA come canale
            tensor_A = [tf.constant(np.array([self.Data_frame_orig.iloc[id].iloc[[j]].values.tolist()[0] for j in range(0, L-1)],dtype=np.float32).T.reshape(2400, L-1), dtype=tf.float32) for id in range(len(self.Data_frame_orig))]            
            tensorAll = tf.stack(tensor_A)
             
        self.tensorAll = tensorAll

        # I am using indices to keep track of the position of the signals in the tensor
        PositionList = [id for id in self.Data_frame_orig.index]

        self.PositionList = PositionList

        return tensorAll, PositionList
    
    def createTensorComb(self, TupleList, GA):
        
        if GA: #se non c'è la s

            sig11 = np.empty(len(TupleList),dtype='object')
            sig21 = np.empty(len(TupleList),dtype='object')
            sig31 = np.empty(len(TupleList),dtype='object')
            sig12 = np.empty(len(TupleList),dtype='object')
            sig22 = np.empty(len(TupleList),dtype='object')
            sig32 = np.empty(len(TupleList),dtype='object')

            #slice the tensor in the APN combinations
            for Indice, combinazione in enumerate(TupleList):
                comb = [self.PositionList.index(combinazione[id]) for id in range(3)]   # posizione, non indice
                sig11[Indice] = tf.gather(self.tensorAll[0], comb[0])
                sig12[Indice] = tf.gather(self.tensorAll[1], comb[0])
                sig21[Indice] = tf.gather(self.tensorAll[0], comb[1])
                sig22[Indice] = tf.gather(self.tensorAll[1], comb[1])
                sig31[Indice] = tf.gather(self.tensorAll[0], comb[2])
                sig32[Indice] = tf.gather(self.tensorAll[1], comb[2])

            sig1 = [tf.stack(sig11), tf.stack(sig12)]
            sig2 = [tf.stack(sig21), tf.stack(sig22)]
            sig3 = [tf.stack(sig31), tf.stack(sig32)]

        else:

            #slice the tensor in the APN combinations
            sig1 = np.empty(len(TupleList),dtype='object')
            sig2 = np.empty(len(TupleList),dtype='object')
            sig3 = np.empty(len(TupleList),dtype='object')

            for Indice, combinazione in enumerate(TupleList):
                comb = [self.PositionList.index(combinazione[id]) for id in range(3)]   # posizione, non indice
                sig1[Indice] = tf.gather(self.tensorAll, comb[0])
                sig2[Indice] = tf.gather(self.tensorAll, comb[1])
                sig3[Indice] = tf.gather(self.tensorAll, comb[2])

            sig1 = [tf.stack(sig1)]
            sig2 = [tf.stack(sig2)]
            sig3 = [tf.stack(sig3)]


        return sig1,sig2,sig3

    # ============================================================================================================================================== 

    def getSiamaseDatasetNoise(self, TupleList, GA, noise = 0.1):

        """
        This function introduces noise to limit overfitting using additional functions (see code below)
        """
    
        # Inject some noise in train to avoid overfitting  (noise is chance of injection)         
        columns = self.Data_frame_orig.columns # Number of columns
        L = len(columns)-2 # Non ho capito cosa sia L

        # If I consider the gestational age
        if GA:

            sig1_1 = [tf.constant(np.array([add_noiseFHR(self.Data_frame_orig.loc[combinazione[0],"FHR"],   noise = noise)] +
                                            ([add_noiseTOCO(self.Data_frame_orig.loc[combinazione[0],"TOCO"], noise = noise)] if "TOCO" in columns else []) +
                                            ([add_noiseFMP(self.Data_frame_orig.loc[combinazione[0],"FMP"], noise = noise)] if "FMP" in columns else []),
                                            dtype=np.float32).T.reshape(2400, L), dtype=tf.float32) for combinazione in TupleList]
            sig1_2 = [self.Data_frame_orig.loc[combinazione[0],"GestAge"] + 0.1 * np.random.randn(1) for combinazione in TupleList]
            sig1_1 = tf.stack(sig1_1)
            sig1_2 = tf.stack(sig1_2)
            sig1 = [sig1_1, sig1_2]

            sig2_1 = [tf.constant(np.array([add_noiseFHR(self.Data_frame_orig.loc[combinazione[1],"FHR"], noise = noise)] + 
                                            ([add_noiseTOCO(self.Data_frame_orig.loc[combinazione[1],"TOCO"], noise = noise)] if "TOCO" in columns else []) +
                                            ([add_noiseFMP(self.Data_frame_orig.loc[combinazione[1],"FMP"], noise = noise)] if "FMP" in columns else []),
                                            dtype=np.float32).T.reshape(2400, L), dtype=tf.float32) for combinazione in TupleList]
            sig2_2 = [self.Data_frame_orig.loc[combinazione[1],"GestAge"] + 0.1 * np.random.randn(1) for combinazione in TupleList]
            sig2_1 = tf.stack(sig2_1)
            sig2_2 = tf.stack(sig2_2)
            sig2 = [sig2_1, sig2_2]

            sig3_1 = [tf.constant(np.array([add_noiseFHR(self.Data_frame_orig.loc[combinazione[2],"FHR"], noise = noise)] + 
                                            ([add_noiseTOCO(self.Data_frame_orig.loc[combinazione[2],"TOCO"], noise = noise)] if "TOCO" in columns else []) +
                                            ([add_noiseFMP(self.Data_frame_orig.loc[combinazione[2],"FMP"], noise = noise)] if "FMP" in columns else []),
                                            dtype=np.float32).T.reshape(2400, L), dtype=tf.float32) for combinazione in TupleList]
            sig3_2 = [self.Data_frame_orig.loc[combinazione[2],"GestAge"] + 0.1 * np.random.randn(1) for combinazione in TupleList]
            sig3_1 = tf.stack(sig3_1)
            sig3_2 = tf.stack(sig3_2)
            sig3 = [sig3_1, sig3_2]

        # If I do not consider the gestational age
        else:

            sig1 = [tf.constant(np.array([add_noiseFHR(self.Data_frame_orig.loc[combinazione[0],"FHR"], noise = noise)] +
                                          ([add_noiseTOCO(self.Data_frame_orig.loc[combinazione[0],"TOCO"], noise = noise)] if "TOCO" in columns else []) +
                                          ([add_noiseFMP(self.Data_frame_orig.loc[combinazione[0],"FMP"], noise = noise)] if "FMP" in columns else []),
                                            dtype=np.float32).T.reshape(2400, L), dtype=tf.float32) for combinazione in TupleList]

            sig2 = [tf.constant(np.array([add_noiseFHR(self.Data_frame_orig.loc[combinazione[1],"FHR"], noise = noise)] +
                                          ([add_noiseTOCO(self.Data_frame_orig.loc[combinazione[1],"TOCO"], noise = noise)] if "TOCO" in columns else []) +
                                          ([add_noiseFMP(self.Data_frame_orig.loc[combinazione[1],"FMP"], noise = noise)] if "FMP" in columns else []),
                                            dtype=np.float32).T.reshape(2400, L), dtype=tf.float32) for combinazione in TupleList]
            
            sig3 = [tf.constant(np.array([add_noiseFHR(self.Data_frame_orig.loc[combinazione[2],"FHR"], noise = noise)] +
                                          ([add_noiseTOCO(self.Data_frame_orig.loc[combinazione[2],"TOCO"], noise = noise)] if "TOCO" in columns else []) +
                                          ([add_noiseFMP(self.Data_frame_orig.loc[combinazione[2],"FMP"], noise = noise)] if "FMP" in columns else []),
                                            dtype=np.float32).T.reshape(2400, L), dtype=tf.float32) for combinazione in TupleList]

            sig1 = tf.stack(sig1)
            sig2 = tf.stack(sig2)
            sig3 = tf.stack(sig3)

        return sig1, sig2, sig3



    # ============================================================================================================================================== 

    def GenerateCombList_reduced(self, limite_settimane, Train = True):

        """
        Generates anchor, positive e negative.
        """

        def IterateCombinations(P,N,limite_settimane,train = True):

            # P is anchor. iterate over L and get one positive and one negative
            
            L1 = len(P)#for the "positive" - the anchor class
            L2 = len(N)#for the "negative" - the other class

            if Train:
                # balance
                L = np.min([L1,L2])
            else:
                L = L1

            # Init list which is used to save
            CombList = []
            
            # Cycle on the index of the lenght of the list
            for i in range(L):

                # Positive anchor
                i_p = random.randint(0,(L1-1)) # Seleziono un indice positivo random nell'array tra 0 e la lunghezza dell'array
                while (abs(P[i,1] - P[i_p,1]) >= limite_settimane) and (i == i_p):
                    i_p = random.randint(0,(L1-1))
                
                # Negative anchor
                i_n = random.randint(0,(L2-1))
                while abs(P[i,1] - N[i_n,1]) >= limite_settimane:
                    i_n = random.randint(0,(L2-1))

                # lowercase p is the final triplet chosen
                p = (P[i,0], P[i_p,0], N[i_n,0])

                # CombList is the final list of triplets
                CombList.append(p)

            return CombList


        GA_df = self.GetSupportdf()
        Class1 = GA_df[GA_df[:,2]==1]
        Class0 = GA_df[GA_df[:,2]==0]
        
        Class1 = Class1[Class1[:,1] < (np.max(Class0[:,1]) + limite_settimane), :]  # Remove rows with values greater than (maxP + limite_giorni - 1)
        Class0 = Class0[Class0[:,1] < (np.max(Class1[:,1]) + limite_settimane), :]
        Class1 = Class1[Class1[:,1] > (np.min(Class0[:,1]) - limite_settimane), :]  
        Class0 = Class0[Class0[:,1] > (np.min(Class1[:,1]) - limite_settimane), :]

        np.random.shuffle(Class1)
        np.random.shuffle(Class0)

        CombList = IterateCombinations(Class1,Class0, limite_giorni, train = Train) + IterateCombinations(Class0, Class1, limite_giorni, train = Train) 
   
        np.random.shuffle(CombList)

        self.CombList = CombList
        return CombList



    # ==============================================================================================================================================   

    def GenerateCombList(self, limite_settimane, Balance = False): #DATI_oggetto,scaler, limite_giorni, Dataset, Train = True, limit = np.inf
    
        # Generate the list of triplet combinations
        # 'limit' can restrict the maximum number of generated combinations
        # Reverse scaling back to the original scale if necessary
        # This is the main function actually used for generating combinations (similar to the other version)

        #inner function for brevity 
        def CreateList():

            GA_df = self.GetSupportdf() #index, GA, label
            
            P = GA_df[GA_df[:,2]==1]
            N = GA_df[GA_df[:,2]==0]
            
            #  ----------------  "anchors" con  i positivi ------------

            # Couple valid with positive
            CombListP = [[P[i1,1], (P[i1,0], P[i2,0])] 
                        for i1 in range(len(P)) for i2 in range(i1 + 1, len(P)) 
                        if np.abs(P[i1,1] - P[i2,1]) < limite_settimane]

            # Add negatives (valid)
            CombList1 = [(*CombListP[i][1], N[j,0]) 
                        for i in range(len(CombListP)) for j in range(len(N)) 
                        if np.abs(CombListP[i][0] - N[j,1]) < limite_settimane]
                
            #  ----------------  "anchors" with the negatives ------------
            CombListN = [[N[i1,1], (N[i1,0], N[i2,0])] 
                        for i1 in range(len(N)) for i2 in range(i1 + 1, len(N)) 
                        if np.abs(N[i1,1] - N[i2,1]) < limite_settimane] 
                
            # Add the positives (valid)
            CombList2 = [(*CombListN[i][1], P[j,0]) 
                        for i in range(len(CombListN)) for j in range(len(P)) 
                        if np.abs(CombListN[i][0] - P[j,1]) < limite_settimane]           

            if Balance:
                #balance
                Lmin = np.min([len(CombList1), len(CombList2)])
                random.shuffle(CombList1)
                random.shuffle(CombList2)
                CombList2 = CombList2[:Lmin]
                CombList1 = CombList1[:Lmin]
                CombList = CombList1 + CombList2
            else:
                CombList = CombList1 + CombList2

            random.shuffle(CombList)

            self.CombList = CombList    
                            
            return CombList
        
        CombList = CreateList()

        return CombList
    
    

    # ============================================================================================================================================== 

    def GenerateCoupleList(self, reset_index = True):
        
        # Generates the list of valid couples (to test the model)

        df = self.Data_frame_orig.copy()

        if reset_index:
            df.reset_index(inplace=True)

        # Make the index a column
        df['ID'] = df.index
        
        P = df[df['label']==1] 
        N = df[df['label']==0]
        
        # Couples PN and NP (list of tuples)
        CombListPN = [(P.iloc[i1]['ID'], N.iloc[i2]['ID']) for i1 in range(len(P)) for i2 in range(len(N))]
        
        # Couples PP and NN (list of tuples)
        CombListPP = [(P.iloc[i1]['ID'], P.iloc[i2]['ID']) for i1 in range(len(P)) for i2 in range(i1 + 1, len(P))]
        CombListNN = [(N.iloc[i1]['ID'], N.iloc[i2]['ID']) for i1 in range(len(N)) for i2 in range(i1 + 1, len(N))]

        return CombListPN, CombListPP + CombListNN        
        

    # ============================================================================================================================================== 
    
    def GetSupportdf(self):
        GA = self.Data_frame_orig.apply(lambda row: extract_first_element(row,"GestAge"), axis=1) #in case i decide to pass is to the network as a vector
        GA_df = pd.DataFrame(GA.to_list(), columns=['Index', 'GA'])  
        GA_df["GA"]=self.scaler.inverse_transform(np.array(GA_df["GA"]).reshape(-1, 1))
        GA_df['GA'] = GA_df['GA'].astype(np.int16)
        GA_df["label"] = self.Data_frame_orig["label"].tolist()
        GA_df['label'] = GA_df['label'].astype(np.int16)
        GA_df['Index'] = GA_df['Index'].astype(np.int16)
        return GA_df.to_numpy()    












# ==========================================================================
# ========================== OTHER USEFUL FUNCITONS ========================
# ==========================================================================

def add_noiseFHR(FHR_scaled, noise = 0.1, scala=0.1, factor=0):

    
    #chances of injection are "noise"
    if random.random() > noise:
        return FHR_scaled   
    else:  
        #inject "red" noise in the signal (it is white when factor = 0)
        x = scala * np.std(FHR_scaled) * np.random.randn(2400)
        y = np.zeros_like(x)
        if factor>0:
            for i in range(1, len(x)):
                y[i] = factor * y[i - 1] + x[i]
        else:
            y = x

        y = y + FHR_scaled
        #add random linear interpolations
        Start = np.random.randint(1, 2400)
        L = np.random.randint(1,7*120)
        if Start + L < 2400:
            y[Start:Start + L] = np.linspace(y[Start], y[Start + L], L).reshape(-1) # Reshape y to have shape (2400,)
            
    return y

def add_noiseTOCO(TOCO_scaled, noise = 0.1):
    
    #chances of injection are "noise"
    if random.random() > noise:
        return TOCO_scaled   
    else:  
        #inject "white" noise in the signal
        y = TOCO_scaled + 0.2 * np.random.randn(2400)
    return y

def add_noiseFMP(FMP, noise = 0.1):
    
    #chances of injection are "noise"
    if random.random() > noise:
        return FMP   
    else:  
        #randomly switch 0 to 1 and viceversa
        random_indices = np.random.choice(2400, int(0.01*2400), replace=False)
        FMP[random_indices] = np.random.choice([0, 1], size=len(random_indices))

    return FMP