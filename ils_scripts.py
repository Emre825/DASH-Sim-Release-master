'''!
@brief This file contains few functions required for IL-scheduler.
'''

import common 
import os 
import numpy as np 

def get_cluster(resource) :

    resource_cluster = 0

    for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :
        ## Skip processing for memory
        if cluster.name == 'MEMORY' :
            continue
        ## if cluster.name == 'MEMORY' :
        if resource in cluster.PE_list :
            resource_cluster = cluster_index
            break
        ## if resource in cluster.PE_list :
    ## for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :

    return resource_cluster

def get_normalized_list(input_list,scheduler = 'IL') :
    
    ## Convert list to numpy array
    np_input_list = np.array(input_list)

    ## Check for list of all zeros
    if len(np_input_list[np_input_list != 0]) == 0 :
        return input_list
    ## if len(np_input_list[np_input_list != 0]) == 0 :

    ## Exclude anomalous values
    if scheduler == 'IL':
        valid_value_list = np_input_list[np_input_list != 10000]
    elif scheduler == 'DAS':
        valid_value_list = np_input_list[np_input_list != 10000000]
    else:
        valid_value_list = np_input_list[np_input_list != 10000]

    ## Check exception and return if list is empty
    if len(valid_value_list) == 0 :
        valid_value_list = np.ones(len(input_list) * 10)
        return valid_value_list
    ## if len(valid_value_list) :

    ## Get min and range of values in list
    min_value_list = np.min(valid_value_list)
    range_value_list = np.max(valid_value_list) - np.min(valid_value_list)

    ## Check exception and normalize
    if range_value_list == 0 :
        normalized_list = np_input_list / min_value_list
    else :
        normalized_list = (np_input_list - min_value_list) / range_value_list
    ## if range_value_list == 0 :

    ## If value is greater than one
    normalized_list[normalized_list > 1] = 10

    return normalized_list

## def get_normalized_list(input_list) :

def das_print_file_headers(filename, ClusterManager) :
    # function to generate DAS dataset file and titles of columns
    file_handle = open(filename, 'w')

    file_handle.write('Time,')
    file_handle.write('TaskID,')

    for value in range(5) :
        file_handle.write('Pred' + str(value) + '_ID,')
    ## for value in range(5) :

    for value in range(5) :
        file_handle.write('Pred' + str(value) + '_CommVol,')
    ## for value in range(5) :

    file_handle.write('NormTaskID,')

    for cluster_index, cluster in enumerate(ClusterManager.cluster_list) :
        if 'MEM' not in cluster.name and 'CAC' not in cluster.name :
            file_handle.write('ExecTime_Cluster' + str(cluster_index) + ',')
    ## for cluster_index, cluster in enumerate(ClusterManager.cluster_list) :

    file_handle.write('DownwardDepth,')
    file_handle.write('RelativeJobID,')
    file_handle.write('JobType,')

    for value in range(5) :
        file_handle.write('Pred' + str(value) + '_Cluster,')
    ## for value in range(5) :
        
    for cluster_index, cluster in enumerate(ClusterManager.cluster_list) :
        if 'MEM' not in cluster.name and 'CAC' not in cluster.name :
            file_handle.write('Cluster' + str(cluster_index) + '_Power,')
    ## for cluster_index, cluster in enumerate(ClusterManager.cluster_list) :    

    for cluster_index, cluster in enumerate(ClusterManager.cluster_list) :
        if 'MEM' not in cluster.name and 'CAC' not in cluster.name :
            file_handle.write('Cluster' + str(cluster_index) + '_FreeTime,')
    ## for cluster_index, cluster in enumerate(ClusterManager.cluster_list) :

    PE_index = 0
    for cluster_index, cluster in enumerate(ClusterManager.cluster_list) :
        if 'MEM' not in cluster.name and 'CAC' not in cluster.name :
            numPE_in_cluster = len(ClusterManager.cluster_list[cluster_index].PE_list)

            for index in range(numPE_in_cluster) :
               file_handle.write('PE' + str(PE_index) + '_FreeTime,')
               PE_index += 1
    ## for cluster_index, cluster in enumerate(ClusterManager.cluster_list) :

    for value in range(9) :
        file_handle.write('ShiftReg_' + str(value) + ',')
    ## for value in range(16) :
    file_handle.write('InjRate8ms' + ',')
    file_handle.write('InjRateTimeDiffus' + ',')    
    file_handle.write('Label\n')

    file_handle.close()
## def das_print_file_headers() :

def ils_print_file_headers(resource_type, filename, num_clusters, num_PEs) :

    file_handle = open(filename, 'w')

    file_handle.write('Time,')
    file_handle.write('TaskID,')
    for value in range(num_PEs) :
        file_handle.write(resource_type + str(value) + '_FreeTime,')
    ## for value in range(num_PEs) :
    file_handle.write('NormTaskID,')
    for value in range(num_clusters) :
        file_handle.write('ExecTime_' + resource_type + str(value) + ',')
    ## for value in range(num_clusters) :
    file_handle.write('DownwardDepth,')
    file_handle.write('RelativeJobID,')
    file_handle.write('JobType,')
    for value in range(5) :
        file_handle.write('Pred' + str(value) + '_ID,')
    ## for value in range(5) :
    for value in range(5) :
        file_handle.write('Pred' + str(value) + '_Cluster,')
    ## for value in range(5) :
    for value in range(5) :
        file_handle.write('Pred' + str(value) + '_Comm,')
    ## for value in range(5) :
    file_handle.write('PE_label,Cluster_label\n')

    file_handle.close()

## def ils_print_file_headers() :

def ils_setup(scale_values) :

    common.dagger_string = ''

    scale_values = str(common.scale_values_list[0]) + '-' + str(common.scale_values_list[0] + 1) + '-1'

    common.model_dagger_string = ''
    if common.ils_dagger_iter != '' :
        if int(common.ils_dagger_iter) != 1 :
            common.model_dagger_string = '_dagger' + str(int(common.ils_dagger_iter) - 1)
        common.dagger_string = '_dagger' + str(int(common.ils_dagger_iter))

    if common.ils_enable_dataset_save or common.ils_enable_dagger :
        os.makedirs('./reports', exist_ok=True)
        os.makedirs('./datasets', exist_ok=True)

## def ils_open_file_handles() :

def open_report_file_handles(scheduler, scale_values) :

    common.dagger_string = ''
    common.model_dagger_string = ''

    name_string = ''
    if common.ils_enable_dataset_save :
        name_string = '_Oracle'
    else :
        name_string = '_IL'
    ## if common.ils_enable_dataset_save :

    if common.ils_dagger_iter != '' :
        if int(common.ils_dagger_iter) != 1 :
            common.model_dagger_string = '_dagger' + str(int(common.ils_dagger_iter) - 1)
        common.dagger_string = '_dagger' + str(int(common.ils_dagger_iter))

    if common.ils_enable_dataset_save or common.ils_enable_dagger :
        os.makedirs('./reports', exist_ok=True)
        os.makedirs('./datasets', exist_ok=True)

    # Create file handles
    report_filename = './reports/report_' + scheduler + name_string + '_' + scale_values + common.model_dagger_string + '.rpt'
    common.report_fp = open(report_filename, 'w')