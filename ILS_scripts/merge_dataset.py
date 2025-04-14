#!/usr/bin/python

import os, re, glob, sys
sys.path.append(os.getcwd())

import common
import DASH_Sim_utils
import configparser
import DASH_SoC_parser
import DTPM_utils


## Command-line arguments for dagger iterations
dagger_iterations = 0

if len(sys.argv) > 1 :
    dagger_iterations = int(sys.argv[1])
## if len(sys.argv) > 1 :

## Specify and read config file
config = configparser.ConfigParser()
config.read('config_file.ini')

## Specify and read SoC file
resource_matrix = common.ResourceManager()
resource_file = "config_SoC/" + config['DEFAULT']['resource_file']
DASH_SoC_parser.resource_parse(resource_matrix, resource_file)

## Define the clusters for which IL policies need to be generated
models = ['clustera']
for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :
    ## Skip processing for memory
    if cluster.name == 'MEMORY' :
        continue
    ## if cluster.name == 'MEMORY' :

    models.append('cluster' + str(cluster_index))
## for cluster_index in range(len(common.ClusterManager.cluster_list) - 1) :

## Iterate through all the clusters
for model in models :

    ## Get a list of dataset files for each cluster
    if dagger_iterations != 0 :
        files = glob.glob('./datasets/data*' + model + '*merged.csv')

        for dagger_iter in range(1, dagger_iterations + 1, 1) :
            files.extend(glob.glob('./datasets/data*' + model + '*1_dagger' + str(dagger_iter) + '*'))
    else :
        files = glob.glob('./datasets/data*' + model + '*-1.csv')
    ## if dagger_iterations != 0 :

    ## Extract the header information of the dataset
    with open(files[0], 'r') as file :
        header = file.readline()
    ## with open(files[0], 'r') as file :

    ## Concatenate multiple files into merged dataset

    if dagger_iterations == 0 :
        output_filename = './datasets/data_IL_' + model + '_merged.csv'
    else :
        output_filename = './datasets/data_IL_' + model + '_merged_dagger' + str(dagger_iterations) + '.csv'
    ## if dagger_iterations == 0 :

    output_filehandle = open(output_filename, 'w')
    output_filehandle.write(header)
    ## Iterate through all files
    for file in files :
        file_handle = open(file, 'r')

        ## Parse each file 
        for line in file_handle :
            if 'Time' in line :
                continue
            output_filehandle.write(line)
        ## for line in file_handle :

        file_handle.close()
    ## for file in files :
    output_filehandle.close()
