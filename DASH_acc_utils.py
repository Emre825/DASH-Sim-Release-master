'''!
@brief This file contains the functions to manage the execution and configuration of accelerators, including DAP.
'''
import copy

import common

def add_config_acc(PE, resource, task):
    '''!
    Add the required configuration to the execution list of the accelerators, and if required, reconfigure the DAP by removing the configurations that are not being used anymore.
    @param PE: PE object to be evaluated
    @param resource: Resource object for this PE
    @param task: Task object
    '''
    if 'DAP' in PE.name:
        DAP_info = get_DAP_info(resource, task)
        task_index = resource.supported_functionalities.index(task.name)
        DAP_configuration = resource.DAP_config_list[task_index]
        cluster = common.ClusterManager.cluster_list[PE.cluster_ID]
        # Update the DAP configuration (if it's not already configured)
        if DAP_configuration not in cluster.current_acc_configuration or \
                (DAP_configuration in cluster.current_acc_configuration and DAP_configuration in cluster.current_acc_kernels_exec):
            # Remove old configurations if reconfiguration is required
            if common.ClusterManager.cluster_list[PE.cluster_ID].DAP_utilization_config + DAP_info.DAP_subPEs > common.DAP_capacity:
                common.ClusterManager.cluster_list[PE.cluster_ID].current_acc_configuration = copy.deepcopy(common.ClusterManager.cluster_list[PE.cluster_ID].current_acc_kernels_exec)
                common.ClusterManager.cluster_list[PE.cluster_ID].DAP_utilization_config = common.ClusterManager.cluster_list[PE.cluster_ID].DAP_utilization_exec
            # Add new configuration
            common.ClusterManager.cluster_list[PE.cluster_ID].current_acc_configuration.append(resource.DAP_config_list[task_index])
            common.ClusterManager.cluster_list[PE.cluster_ID].DAP_utilization_config += DAP_info.DAP_subPEs
            # Update the reconfiguration overhead
            common.ClusterManager.cluster_list[PE.cluster_ID].reconfiguration_overhead = max(DAP_info.programming_latency, common.ClusterManager.cluster_list[PE.cluster_ID].reconfiguration_overhead)
        # Update the kernels that are being executed
        common.ClusterManager.cluster_list[PE.cluster_ID].current_acc_kernels_exec.append(resource.DAP_config_list[task_index])
        # Update utilization
        common.ClusterManager.cluster_list[PE.cluster_ID].DAP_utilization_exec += DAP_info.DAP_subPEs
    elif PE.type == 'ACC':
        task_index = resource.supported_functionalities.index(task.name)
        common.ClusterManager.cluster_list[PE.cluster_ID].current_acc_kernels_exec.append(resource.DAP_config_list[task_index])

def remove_config_current_exec_acc(PE, resource, task):
    '''!
    After executing the task, remove it from the ACC execution list.
    @param PE: PE object to be evaluated
    @param resource: Resource object for this PE
    @param task: Task object
    '''
    if 'DAP' in PE.name:
        DAP_info = get_DAP_info(resource, task)
        task_index = resource.supported_functionalities.index(task.name)
        common.ClusterManager.cluster_list[PE.cluster_ID].current_acc_kernels_exec.remove(resource.DAP_config_list[task_index])
        common.ClusterManager.cluster_list[PE.cluster_ID].DAP_utilization_exec -= DAP_info.DAP_subPEs
    elif PE.type == 'ACC':
        task_index = resource.supported_functionalities.index(task.name)
        common.ClusterManager.cluster_list[PE.cluster_ID].current_acc_kernels_exec.remove(resource.DAP_config_list[task_index])

def get_DAP_info(resource, task):
    '''!
    Get the DAP profile for a given type of task
    @param resource: Resource to be evaluated
    @param task: Task object
    '''
    acc_ID = common.ClusterManager.cluster_list[resource.cluster_ID].name + ',' + resource.DAP_config_list[resource.supported_functionalities.index(task.name)]
    acc_info = common.resource_matrix_Acc.dict[acc_ID]
    return acc_info

def get_DAP_reconfiguration_overhead(PE, resource, task):
    '''!
    Get the reconfiguration overhead for the DAP if the configuration is not currently programmed.
    @param PE: PE object to be evaluated
    @param resource: Resource object for this PE
    @param task: Task object
    '''
    if 'DAP' in PE.name:
        task_index = resource.supported_functionalities.index(task.name)
        DAP_configuration = resource.DAP_config_list[task_index]
        cluster = common.ClusterManager.cluster_list[PE.cluster_ID]
        if DAP_configuration in cluster.current_acc_configuration:
            return 0
        else:
            DAP_info = get_DAP_info(resource, task)
            return DAP_info.programming_latency
    else:
        return 0

