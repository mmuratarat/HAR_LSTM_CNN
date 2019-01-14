import os
import pandas as pd
import numpy as np
import math

def reading_data(data_path, split = "train"):
    
    n_class = 6
    n_steps = 128
    
    #Paths
    path = os.path.join(data_path, split)
    path_to_signals = os.path.join(path, "Inertial_Signals")
    
    #Reading labels
    labels_path = os.path.join(path, "y_" + split + ".txt")
    labels = pd.read_csv(labels_path, header = None)
    
    channel_files = os.listdir(path_to_signals)
    channel_files.sort()
    channel_files.remove('.ipynb_checkpoints')
    n_channels = len(channel_files)
    # -1 because '.ipynb_checkpoints' might be in the folder.
    
    posix = len(split) + 5
    
    # Initiate array
    list_of_channels = []
    
    X = np.zeros((len(labels), n_steps, n_channels))
    idx_channel = 0
    for fil_ch in channel_files:
        channel_name = fil_ch[:-posix]
        dat_ = pd.read_csv(os.path.join(path_to_signals,fil_ch), delim_whitespace = True, header = None, engine='python')
        X[:,:,idx_channel] = dat_.values
        
        # Record names
        list_of_channels.append(channel_name)
        
        idx_channel += 1

    # Return 
    return X, labels[0].values, list_of_channels

def one_hot(y, num_classes):
    return (np.squeeze(np.eye(num_classes)[y.reshape(-1)-1])).astype(np.int32)

#Standardize the data based on mean and std of each feature in each time-step.
def standardize(train, test):
	""" Standardize data """

	# Standardize train and test
	X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
	X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]

	return X_train, X_test
#Example:
#This finds the average of the first channel (feature) in first time-step over all the 7352 observations
#average = 0
#for i in range(7352):
#    average += xTrain[i,0,0]/7352
#print(average)

def get_batches(X, y, batch_size = 100):
	""" Return a generator for batches """
	n_batches = len(X) // batch_size
	X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

	# Loop over batches and yield
	for b in range(0, len(X), batch_size):
		yield X[b:b+batch_size], y[b:b+batch_size]
        
def miniBatch(x, y, batchSize):
    numObs  = x.shape[0]
    batches = [] 
    batchNum = math.floor(numObs / batchSize)

    for i in range(batchNum):
        xBatch = x[i * batchSize:(i + 1) * batchSize, :]
        yBatch = y[i * batchSize:(i + 1) * batchSize, :]
        batches.append((xBatch, yBatch))
    xBatch = x[batchNum * batchSize:, :]
    yBatch = y[batchNum * batchSize:, :]
    batches.append((xBatch, yBatch))
    return batches