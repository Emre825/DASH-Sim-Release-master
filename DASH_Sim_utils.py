'''!
@brief This file contains functions that are used by DASH_Sim.
'''

import os
import csv
import fnmatch
import sys

import common
import DASH_acc_utils

trace_list = [common.TRACE_FILE_SYSTEM, common.TRACE_FILE_TASKS, common.DATASET_FILE_DTPM, common.TRACE_FILE_FREQUENCY, common.TRACE_FILE_PES, common.TRACE_FILE_TEMPERATURE, common.TRACE_FILE_IL_PREDICTIONS, common.TRACE_FILE_LOAD, common.TRACE_FILE_TEMPERATURE_WORKLOAD]

def check_PE_capacity(PE, resource, task):
    '''!
    Check whether the PE has capacity to execute the given task.
    @param PE: PE to be evaluated
    @param resource: Resource object of the given PE
    @param task: Task to be evaluated
    '''
    if 'DAP' in PE.name:
        DAP_info = DASH_acc_utils.get_DAP_info(resource, task)
        if common.ClusterManager.cluster_list[PE.cluster_ID].DAP_utilization_exec + DAP_info.DAP_subPEs <= common.DAP_capacity:
            return (len(PE.queue) < PE.capacity)
        else:
            return False
    else:
        return (len(PE.queue) < PE.capacity)

def update_PE_utilization_and_info(PE, current_timestamp):
    '''!
    Update the PE utilization.
    @param PE: PE to be evaluated
    @param current_timestamp: Current timestamp
    '''
    lower_bound = current_timestamp-common.sampling_rate             # Find the lower bound for the time window_list under consideration

    completed_info = []
    running_info = []
    for task in common.TaskQueues.completed.list:
        #print('Time %s:'%current_timestamp, task.start_time, task.finish_time, task.PE_ID)
        if task.PE_ID == PE.ID:
            if ((task.start_time < lower_bound) and (task.finish_time < lower_bound)):
                continue
            elif ((task.start_time < lower_bound) and (task.finish_time >= lower_bound)):
                completed_info.append(lower_bound)
                completed_info.append(task.finish_time)
            else:
                completed_info.append(task.start_time)
                completed_info.append(task.finish_time)
            #print('Time %s:'%current_timestamp,'completed',completed_info, 'Task', task.ID, 'PE', PE.ID)

    for task in common.TaskQueues.running.list:
        #print('Time %s:'%current_timestamp, task.start_time, task.PE_ID)
        if task.PE_ID == PE.ID:
            if (task.start_time < lower_bound):
                running_info.append(lower_bound)
            else:
                running_info.append(task.start_time)
                task_start_time = task.start_time
            running_info.append(current_timestamp)
            #print('Time %s:'%current_timestamp,'running',running_info, 'Task', task.ID, 'PE', PE.ID)
    merged_list = completed_info + running_info
    # get the utilization for the PE
    sum_of_active_times  = sum([merged_list[i*2+1] - merged_list[i*2] for i in range(int(len(merged_list)/2))])
    PE.utilization = (sum_of_active_times/common.sampling_rate) / PE.capacity
    # print(PE.ID, PE.utilization)

    full_list = [PE.ID, PE.utilization, current_timestamp]
    info_list = [0 if i > (len(merged_list)-1) else merged_list[i] for i in range(12)]
    full_list.extend(info_list)

    PE.info = info_list

    # if (common.DEBUG_SIM):
    #     print('Time %s: for PE-%d'%(current_timestamp,PE.ID),PE.info)


def trace_frequency(timestamp):
    '''!
    Trace method for saving the frequency variations.
    @param timestamp: Current timestamp
    '''
    if (common.TRACE_FREQUENCY):
        create_header = False
        if not (os.path.exists(common.TRACE_FILE_FREQUENCY)):
            # Create the CSV header
            create_header = True
        with open(common.TRACE_FILE_FREQUENCY, 'a', newline='') as csvfile:
            trace = csv.writer(csvfile, delimiter=',')
            if create_header == True:
                header_list = ['Timestamp']
                for idx, current_cluster in enumerate(common.ClusterManager.cluster_list):
                    if current_cluster.type != "MEM":
                        header_list.append('f_PE_' + str(idx))
                        header_list.append('N_PE_' + str(idx))
                trace.writerow(header_list)
            data = [timestamp]
            for idx, current_cluster in enumerate(common.ClusterManager.cluster_list):
                if current_cluster.type != "MEM":
                    data.append(current_cluster.current_frequency / 1000)
                    data.append(current_cluster.num_active_cores)
            trace.writerow(data)

def trace_tasks(task, PE, task_time, total_energy):
    '''!
    Trace method for saving statistics about the tasks.
    @param task: Task to be traced
    @param PE: Current PE
    @param task_time: Task's execution time
    @param total_energy: Task's total energy consumption
    '''
    if (common.TRACE_TASKS):
        create_header = False
        if not (os.path.exists(common.TRACE_FILE_TASKS.split(".")[0] + "__" + str(common.trace_file_num) + ".csv")):
            # Create the CSV header
            create_header = True
        with open(common.TRACE_FILE_TASKS.split(".")[0] + "__" + str(common.trace_file_num) + ".csv", 'a', newline='') as csvfile:
            trace = csv.writer(csvfile, delimiter=',')
            if create_header == True:
                trace.writerow(['DVFS policy', 'Task ID', 'PE', 'Exec. Time (us)', 'Energy (J)'])
            trace.writerow([common.ClusterManager.cluster_list[PE.cluster_ID].DVFS, task.ID, common.ClusterManager.cluster_list[PE.cluster_ID].name, task_time, total_energy])

def trace_system():
    '''!
    Trace method for saving statistics related to the system, i.e., the whole simulation.
    '''
    if (common.TRACE_SYSTEM):
        create_header = False
        if not (os.path.exists(common.TRACE_FILE_SYSTEM.split(".")[0] + "__" + str(common.trace_file_num) + ".csv")):
            # Create the CSV header
            create_header = True
        with open(common.TRACE_FILE_SYSTEM.split(".")[0] + "__" + str(common.trace_file_num) + ".csv", 'a', newline='') as csvfile:
            trace = csv.writer(csvfile, delimiter=',')
            if create_header == True:
                trace.writerow(['Job List', 'DVFS mode', 'N_little', 'N_big', 'Exec. Time (us)', 'Cumulative Exec. Time (us)', 'Energy (J)'])
            DVFS_mode_list = []
            for DVFS_config in common.DVFS_cfg_list:
                if DVFS_config == "performance":
                    DVFS_mode_list.append("P")
                elif DVFS_config == "powersave":
                    DVFS_mode_list.append("LP")
                elif DVFS_config == "ondemand":
                    DVFS_mode_list.append("OD")
                elif str(DVFS_config).startswith("constant"):
                    split = str(DVFS_config).split('-')
                    DVFS_mode_list.append("C" + split[1])
            if common.simulation_mode == "validation":
                trace.writerow([common.current_job_list, DVFS_mode_list, common.gen_trace_capacity_little, common.gen_trace_capacity_big,
                                common.results.execution_time, common.results.execution_time, common.results.energy_consumption])
            elif common.simulation_mode == "performance":
                if len(common.job_list) == 1:
                    job_list = common.current_job_list
                else:
                    job_list = common.job_list
                trace.writerow([job_list, DVFS_mode_list, common.gen_trace_capacity_little, common.gen_trace_capacity_big,
                                common.results.execution_time - common.warmup_period, common.results.cumulative_exe_time,
                                common.results.cumulative_energy_consumption])

def trace_PEs(timestamp, PE):
    '''!
    Trace method for saving statistics related to the PEs.
    @param timestamp: Current timestamp
    @param PE: PE to be traced
    '''
    if (common.TRACE_PES):
        create_header = False
        if not (os.path.exists(common.TRACE_FILE_PES)):
            # Create the CSV header
            create_header = True
        with open(common.TRACE_FILE_PES, 'a', newline='') as csvfile:
            dataset = csv.writer(csvfile, delimiter=',')
            if create_header == True:
                dataset.writerow(['Timestamp', 'PE', 'Info'])
            dataset.writerow([timestamp, PE.ID, PE.info])

def trace_IL_predictions(timestamp, freq_pred, num_cores_pred=[]):
    '''!
    Trace method for saving the DTPM IL predictions.
    @param timestamp: Current timestamp
    @param freq_pred: Predicted frequency by the IL policy
    @param num_cores_pred: Predicted number of active cores by the IL policy
    '''
    if (common.TRACE_FILE_IL_PREDICTIONS):
        create_header = False
        if not (os.path.exists(common.TRACE_FILE_IL_PREDICTIONS)):
            # Create the CSV header
            create_header = True
        with open(common.TRACE_FILE_IL_PREDICTIONS, 'a', newline='') as csvfile:
            dataset = csv.writer(csvfile, delimiter=',')
            if create_header == True:
                header_list = ['Timestamp']
                for idx, current_cluster in enumerate(common.ClusterManager.cluster_list):
                    if current_cluster.type != "MEM":
                        header_list.append('f_PE_' + str(idx))
                        if num_cores_pred != []:
                            header_list.append('N_PE_' + str(idx))
                dataset.writerow(header_list)
            data = [timestamp]
            for idx, current_cluster in enumerate(common.ClusterManager.cluster_list):
                if current_cluster.type != "MEM":
                    data.append(freq_pred[idx])
                    if num_cores_pred != []:
                        data.append(num_cores_pred[idx])
            dataset.writerow(data)

def trace_temperature(timestamp):
    '''!
    Trace method for saving the temperature variations.
    @param timestamp: Current timestamp
    '''
    if (common.TRACE_TEMPERATURE):
        create_header = False
        if not (os.path.exists(common.TRACE_FILE_TEMPERATURE)):
            # Create the CSV header
            create_header = True
        with open(common.TRACE_FILE_TEMPERATURE, 'a', newline='') as csvfile:
            dataset = csv.writer(csvfile, delimiter=',')
            if create_header == True:
                dataset.writerow(['Timestamp', 'Snippet', 'Temperature', 'Throttling_state'])
            dataset.writerow([timestamp, common.current_job_list, max(common.current_temperature_vector), common.throttling_state])

def trace_load(timestamp, PEs):
    '''!
    Trace method for saving the load variations
    @param timestamp: Current timestamp
    @param PEs: List of PEs
    '''
    if (common.TRACE_LOAD):
        create_header = False
        if not (os.path.exists(common.TRACE_FILE_LOAD)):
            # Create the CSV header
            create_header = True
        with open(common.TRACE_FILE_LOAD, 'a', newline='') as csvfile:
            dataset = csv.writer(csvfile, delimiter=',')
            if create_header == True:
                header_list = ['Timestamp', 'Snippet']
                for idx, current_cluster in enumerate(common.ClusterManager.cluster_list):
                    if current_cluster.type != "MEM":
                        header_list.append('N_tasks_PE_' + str(idx))
                header_list.append('N_tasks_total')
                dataset.writerow(header_list)
            data = [timestamp, common.current_job_list]
            total_num_tasks = 0
            for idx, current_cluster in enumerate(common.ClusterManager.cluster_list):
                if current_cluster.type != "MEM":
                    num_tasks = get_num_tasks_being_executed(current_cluster, PEs)
                    data.append(num_tasks)
                    total_num_tasks += num_tasks
            data.append(total_num_tasks)
            dataset.writerow(data)

def get_system_state(PEs, input_freq=None, input_cores=None):
    '''!
    Get the current system state.
    @param PEs: List of PEs
    @param input_freq: Input frequency for predicting other states (Optional).
    @param input_cores: Input number of cores for predicting other states (Optional).
    @return System state
    '''
    frequency_list = []
    utilization_list = []
    N_big = 0
    N_little = 0
    for current_cluster in common.ClusterManager.cluster_list:
        if current_cluster.type != "MEM":
            if input_freq == None:
                frequency_list.append(current_cluster.current_frequency)
            else:
                frequency_list.append(input_freq[current_cluster.ID] * 1000)
            utilization_list.append(get_cluster_utilization(current_cluster, PEs))
        if current_cluster.type == "LTL":
            if input_cores == None:
                N_little = current_cluster.num_active_cores
            else:
                N_little = input_cores[current_cluster.ID]
        elif current_cluster.type == "BIG":
            if input_cores == None:
                N_big = current_cluster.num_active_cores
            else:
                N_big = input_cores[current_cluster.ID]

    samples = common.hardware_counters[(common.hardware_counters['Job List'] == str(common.current_job_list)) &
                                       (common.hardware_counters['N_little'] == N_little) &
                                       (common.hardware_counters['N_big'] == N_big)]

    current_state = None
    for index, sample in samples.iterrows():
        break_flag = False
        for freq_index, freq in enumerate(frequency_list):
            search_string = "FREQ_PE_" + str(freq_index) + " (GHz)"
            if freq != sample[search_string] * 1000:
                break_flag = True
                break
        if break_flag is False:
            if current_state != None:
                print("[E] Multiple system states match the search query")
                print("[E] {}".format(current_state))
                print("[E] {}".format(sample))
                sys.exit()
            current_state = sample

    if current_state is None:
        print("[E] Could not get current system state from the HW counter CSV. Job list: {}, Frequency: {}, "
              "N_little: {}, N_big: {}".format(common.current_job_list, frequency_list, N_little, N_big))
        sys.exit()

    for index, utilization in enumerate(utilization_list):
        current_state['Utilization_PE_' + str(index)] = utilization

    return current_state

def get_current_job_list():
    '''!
    Get the current snippet.
    @return Current snippet
    '''
    # Get the current job list based on the snippet ID while injecting jobs
    if common.job_list != []:
        return common.job_list[common.snippet_ID_exec]
    else:
        return common.job_list

def get_num_tasks_being_executed(cluster, PEs):
    '''!
    Get the number of tasks that are being executed.
    @param cluster: Current cluster
    @param PEs: List of PEs
    @return Number of tasks being executed
    '''
    # Get the number of tasks currently being executed in the cluster
    num_tasks = 0
    for PE_ID in cluster.PE_list:
        if not PEs[PE_ID].idle:
            num_tasks += 1
    return num_tasks

def get_cluster_utilization(cluster, PEs):
    '''!
    Get the cluster utilization.
    @param cluster: Current cluster
    @param PEs: List of PEs
    @return Cluster utilization
    '''
    # Get the cluster utilization
    utilization = 0
    for PE_ID in cluster.PE_list:
        utilization += PEs[PE_ID].utilization
    return utilization / len(cluster.PE_list)

def clean_traces():
    '''!
    Remove old trace files.
    '''
    for trace_name in trace_list:
        if os.path.exists(trace_name):
            os.remove(trace_name)
    # Remove old traces generated in parallel
    for trace_name in trace_list:
        file_list = fnmatch.filter(os.listdir('.'), trace_name.split(".")[0] + '__*.csv')
        for f in file_list:
            os.remove(f)

def clean_policies():
    '''!
    Remove old policy files.
    '''
    file_list = fnmatch.filter(os.listdir('.'), '*.pkl')
    for f in file_list:
        os.remove(f)

def init_variables_at_sim_start() :
    '''!
    Initialize config variables.
    '''
    for cluster in common.ClusterManager.cluster_list:
        cluster.current_acc_kernels_exec = []
        cluster.current_acc_configuration = []
        cluster.DAP_utilization_exec = 0
        cluster.DAP_utilization_config = 0
        cluster.reconfiguration_overhead = 0
    common.snippet_start_time = common.warmup_period
    common.snippet_ID_inj = -1
    common.snippet_ID_exec = 0
    common.snippet_throttle = -1
    common.snippet_temp_list = []
    common.snippet_initial_temp = [common.T_ambient,
                                   common.T_ambient,
                                   common.T_ambient,
                                   common.T_ambient,
                                   common.T_ambient]
    common.DAgger_last_snippet_ID_freq = -1
    common.DAgger_last_snippet_ID_num_cores = -1
    common.DAgger_last_snippet_ID_regression = -1
    common.current_temperature_vector = [common.T_ambient,  # Indicate the current PE temperature for each hotspot
                                         common.T_ambient,
                                         common.T_ambient,
                                         common.T_ambient,
                                         common.T_ambient]
    common.B_model = []
    common.job_counter_list = [0]*len(common.current_job_list)
    common.aggregate_data_freq = False
    common.aggregate_data_num_cores = False
    common.aggregate_data_regression = False
    common.thermal_limit_violated = False
    common.throttling_state = -1
