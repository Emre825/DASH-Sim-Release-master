# -*- coding: utf-8 -*-
"""
Created on Thu Sept14 13:18:59 2019
"""

import os
import re
import math
import shutil
import random
import scipy.io
import numpy as np
import pandas as pd                
import matplotlib.pyplot as plt
import sklearn
import pickle

from sklearn.feature_selection import chi2,RFECV
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeRegressor      
from sklearn.tree import DecisionTreeClassifier      
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from IPython.display import Image
from sklearn import tree
import pydotplus

from datetime import datetime
from time import sleep

import csv
import random

import itertools

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers
from keras import regularizers
from datetime import datetime
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

import sys
sys.path.append('./')
sys.path.append('./DAS_scripts')

from functions_feature_selection import *
# import run_sims_all_injections_ILS

os.environ["KMP_WARNINGS"] = "FALSE" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED']=str(1000)
import random
random.seed(1000)
from numpy.random import seed
np.random.seed(1000)
import tensorflow as tf
tf.random.set_seed(1000)

##############################################################################

os.makedirs('../reports/', exist_ok=True)

app_name        = "data_IL"
# classifier_types = ['RT', 'SVM', 'LiR', 'LoR', 'NN']
classifier_types = ['RT']

# Depths for regression tree
max_tree_depth_array = [8]

dagger_string = ''

models = ['clustera']

labelings = [1]

classifier_type = classifier_types[0]

## Get all injection rates
# scale_values_list = run_sims_all_injections_ILS.scale_values_list
scales = [0]
# for scale in scale_values_list :
#     values = scale.split('-')
#     scale = values[0]
#     scales.append(scale)
## for scale in scale_values_list :
for labeling in labelings:
    # File handle to print accuracy
    accuracy_filename = './reports/' + classifier_type + dagger_string + '_training_accuracy.rpt'
    accuracy_file = open(accuracy_filename, 'w')
    accuracy_file.close()

    # Use classification / regression
    if classifier_type == 'NN' :
        ml_type = 'classification'
    else :
        ml_type = 'classifier'
        
    # Iterate over each of the models
    for model_index, model in enumerate(models) :
        
        # Create environment with required number of labels
        # num_PEs = num_list[model_index]
        env = ApplicationEnv(1)

        # Specify the name of the file
        datafile_name = './datasets/DAS/postprocessed/data_DAS_merged_all_postprocessed' + ".csv"

        # Ignore a model if it is has no data
        num_lines_in_datafile = len(open(datafile_name).readlines())
        if num_lines_in_datafile < 5 :
            continue
        
        # Read dataset into Pandas dataframe
        alldata_orig = pd.read_csv("./" + datafile_name,header=0,engine='python')
        print('Dataset imported...')
        alldata_orig = alldata_orig.dropna()
        # Extract feature data and labels
        num_features = alldata_orig.shape[1] - 3

        features = list(range(2,67))
        features.append(76)

        feature_data = alldata_orig.iloc[:,features]
        labels = alldata_orig.iloc[:, -1]
        
        # Check if there are more than one unique labels
        if len(np.unique(labels)) == 1 :
            print(model + " " + "accuracy: " + str(100.0))
            
            accuracy_file = open(accuracy_filename, 'a')
            accuracy_file.write(model + " " + "accuracy: " + str(100.0) + '\n')
            accuracy_file.close()
            continue
              
        # Separate test and train data from the given dataset
        train_size = 0.7
        test_size  = 0.3

        feature_data_train, feature_data_test, labels_train, labels_test = \
        train_test_split(feature_data, labels, test_size=test_size, train_size=train_size, random_state=0)

        ## Complex logic to find the number of samples belonging to each injection rate
        inj_rate_indices = [0]
        inj_rate_samples = []
        inj_rate_counter = 0
        num_samples_counter = 0
        prev_time = 0
        feature_data_test = feature_data_test.sort_index()
        labels_test = labels_test.sort_index()

        # Iterate for each tree depth of decision tree
        for max_tree_depth in max_tree_depth_array:
        
            # Specify filename to save model
            if classifier_type == "NN":
                filename = './models/' + classifier_type + "_" + model + "_" + dagger_string + "_model.h5"
            else:
                filename = './models/all_' + str(max_tree_depth) +'_'+ classifier_type + "_" + model + ".sav"

            # Training phase
            phase = "training"
        
            # Train model with train features and labels
            ml_model = env.f_train_model(feature_data_train, labels_train, classifier_type, ml_type, max_tree_depth)
            os.makedirs('./models/', exist_ok=True)
            # Save model
            if classifier_type == "NN":
                ml_model.save(filename)
            else:
                pickle.dump(ml_model, open(filename, 'wb'))

            # Testing phase                
            phase = "testing"
            
            # Load model
            if classifier_type == "NN":
                regressor = load_model(filename)
            else:
                regressor = pickle.load(open(filename, 'rb'))        
            
            # Test model and calculate accuracy
            accuracy, output_labels = env.f_test_model(classifier_type, ml_type, feature_data_test, regressor, labels_test)    

            print(classifier_type + ' policy for %s accuracy: %.2f%%' %(model, accuracy*100))
            # for i in range(len(scales)):
            #     print(classifier_type + ' policy for scale %s accuracy: %.2f%%' %(scales[i], scale_accuracy[i]*100))
            
            accuracy_file = open(accuracy_filename, 'a')
            accuracy_file.write(model + " " + "accuracy: " + str(accuracy*100) + '\n')
            accuracy_file.close()

