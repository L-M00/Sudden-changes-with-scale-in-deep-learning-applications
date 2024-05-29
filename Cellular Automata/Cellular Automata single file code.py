# This file contains the code for the cellular automata test, using infinite data.

# It is being combined into a single file with the intention of being run of the Hydra compute cluster.

# Importing necessary libraries

#!pip install cellpylib

import cellpylib as cpl
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
#import seaborn as sns
#import time

# Data generation parameters
data_size = 100 # the number of data points in each row of data
#programmes_considered = np.arange(0,256,1) # the set of programmes being considered. For the 1D case it makes sense to consider all 0 to 255 programmes.
#number_of_samples = 2000 # the number of random times the output of a programme will be calculated, given random inputs
timesteps = 100 # the number of timesteps which each programme is run for before the output is used to train the model

# Training parameters
num_epochs = 5000  # Number of training epochs
hidden_size = 256  # Update with the desired size of the hidden layer
learning_rate = 0.001 # learning rate used later in the optimizer
batch_size = 512 # Batch size used when creating the train and test datasets. Note that 5 is likely much too low, and 32 would be more suitable for this problem.
train_ratio = 0.8 # Specifies how much of the set will be used to training vs testing

# Programme distribution

programmes_prob_distribution = []
for i in range(256):
    programmes_prob_distribution.append((i+10)**(-1))
programmes_prob_distribution = np.array(programmes_prob_distribution) 
# Note that this distribution will be normalised inside the data pre-processing step if not already normalised here

# Model Initialisation / Training setup

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.bn2 = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        return out

# Define the input size, hidden size, and number of classes
input_size = data_size  # Update with the actual input size
#hidden_size = 64  # Update with the desired size of the hidden layer
#num_classes = len(programmes_considered)+1  # Number of potential classes
num_classes = 256 #Number of potential classes, here stuck at 256

# Create an instance of the neural network
model = NeuralNetwork(input_size, hidden_size, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Data generation functions (where the programmes_considered have a probability distribution)

def create_data(data_size, programmes_prob_distribution, number_of_samples, timesteps):

    # Creating the dataset and labels variables to be populated later
    dataset = np.empty(shape=(number_of_samples, data_size), dtype=int) # each row is data_size length, with number_of_samples rows
    labels = np.empty(shape=(1, number_of_samples), dtype=int)

    # Stating the space of considered programmes
    programmes = np.arange(0,256,1)

    # Normalising the distribution in case it is not already normalised
    programmes_total = sum(programmes_prob_distribution)
    programmes_prob_distribution_norm = [x / programmes_total for x in programmes_prob_distribution]
    
    for i in range(number_of_samples):

        # Randomly selecting a rule number according to the probability distribution given
        rule_number = np.random.choice(a = programmes, size=None, replace=True, p = programmes_prob_distribution_norm)
        #print(f"Considering rule_number = ", rule_number)
        cellular_automaton = cpl.init_random(data_size)
        cellular_automaton = cpl.evolve(cellular_automaton, timesteps=timesteps, memoize=True, apply_rule=lambda n, c, t: cpl.nks_rule(n, rule_number))
        #print(cellular_automaton[-1])
        dataset[i] = cellular_automaton[-1]
        labels[:,i] = rule_number

    return [dataset, labels]


def data_split(data, train_ratio):

    np.random.shuffle(data) #randomly select parts of the dataset
    #train_ratio = train_ratio # this reserves 80% for training, 20% for testing
    split_index = int(len(data) * train_ratio)
    
    train_data = data[:split_index]
    test_data = data[split_index:]
    #print(f"train_data = ", train_data)
    #print(f"test_data = ", test_data)
    
    # Separate the dataset and labels from the training and testing sets
    train_dataset, train_labels = zip(*train_data)
    test_dataset, test_labels = zip(*test_data)
    
    data_split = [train_dataset, train_labels, test_dataset, test_labels]
    return data_split

def data_loader(data_size, programmes_prob_distribution, number_of_samples, timesteps, train_ratio):

    # Generate the data according to input parameters
    [dataset, labels] = create_data(data_size, programmes_prob_distribution, number_of_samples, timesteps)
    labels = labels[0] # Deal with the fact that the output is a list of a single list

    # Shifting the labels such that they are indexed from 0. Required for cross entropy to work
    #labels = [x - min(labels) for x in labels] #!!! Not currently shifting labels in a test to alter them later - may help with training in smaller batches
    # Use data_split
    data = [(data_sample, label) for data_sample, label in zip(dataset, labels)]
    [train_dataset, train_labels, test_dataset, test_labels] = data_split(data, train_ratio)

    tensor_train_dataset = TensorDataset(Tensor(train_dataset), Tensor(train_labels))
    tensor_test_dataset = TensorDataset(Tensor(test_dataset), Tensor(test_labels))
    
    train_loader = DataLoader(tensor_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tensor_test_dataset, batch_size=batch_size, shuffle=True)

    return [train_loader, test_loader]

# Training loop (includes data generation)

def main_train(data_size, programmes_prob_distribution, batch_size, timesteps, train_ratio, num_epochs):

    # Initisalise training and test loss tracking variables
    training_loss = np.empty(num_epochs)
    test_loss = np.empty(num_epochs)
    
    # Each epoch here trains over 1 batch size of data (which at the moment is 32). Each epoch is therefore smaller and better controlled.
    for epoch in range(num_epochs):
        [train_loader, test_loader] = data_loader(data_size, programmes_prob_distribution, batch_size, timesteps, train_ratio)
        for data, labels in train_loader:
            # Forward pass
            outputs = model(data)
    
            #Shifting labels for loss calculation
            shifted_labels = labels - torch.min(labels)
            shifted_labels = shifted_labels.long()
            loss = criterion(outputs, shifted_labels)
            
            # monitoring test loss during training
            for data, labels in test_loader:
                labels_test = labels.long()
                outputs_test = model(data)
                loss_test = criterion(outputs_test, labels_test)
                test_loss[epoch] = loss_test.item()
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # Print the loss after each epoch
        #if epoch%10==0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
        training_loss[epoch] = loss.item()

    return training_loss, test_loss

training_loss, test_loss = main_train(data_size, programmes_prob_distribution, batch_size, timesteps, train_ratio, num_epochs)

