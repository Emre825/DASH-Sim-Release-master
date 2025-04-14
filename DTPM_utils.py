'''!
@brief This file contains functions that are used by the DTPM module.
'''

import pandas as pd
import multiprocessing
import itertools
import numpy as np
from itertools import islice
import sys
import math
# matplotlib.use('Agg')
import os
from sys import platform
from statistics import mean
import csv
import copy
import re, shutil, tempfile

import DTPM_power_models
import common
import DASH_Sim_v0
import DASH_Sim_utils
import DTPM_policies

def run_sim_initial_dataset(sim_params):
    '''!
    Call the simulator with specific parameters in order to generate the dataset that will be used for the DTPM policies.
    This method is called multiple times in parallel, based on the availability of CPU cores in the host machine.
    Each simulation executes and obtains the results from a single snippet (one sample in the dataset).
    @param sim_params: Simulation configurations that change among different simulations, for instance, DVFS config and snippet.
    '''
    sim_num, DVFS_cfg_list, job_config, N_little, N_big, num_PEs = sim_params
    common.CLEAN_TRACES = False
    common.generate_complete_trace = True
    common.TRACE_PES = False
    common.TRACE_IL_PREDICTIONS = False
    common.TRACE_TEMPERATURE = False
    common.enable_real_time_constraints = False
    common.enable_num_cores_prediction = False
    common.DVFS_cfg_list = DVFS_cfg_list
    common.job_list = job_config
    if job_config == []:
        common.current_job_list = []
    else:
        common.current_job_list = job_config[0]
        common.snippet_size = sum(job_config[0])
        common.max_num_jobs = common.snippet_size * len(job_config)
    common.gen_trace_capacity_little = N_little
    common.gen_trace_capacity_big = N_big
    common.sim_ID = sim_num
    common.num_PEs_TRACE = num_PEs

    common.trace_file_num = os.getpid()
    DASH_Sim_v0.run_simulator()

def run_parallel_sims(DVFS_config_list, N_little_list, N_big_list, N_jobs, N_applications, heterogeneous_PEs=False):
    '''!
    Run all possible combinations for the provided frequency points in parallel.
    Create the simulation parameters for all simulations.
    @param DVFS_config_list: List of all DVFS configurations to be executed
    @param N_little_list: List of all number of little cores to be executed
    @param N_big_list: List of all number of big cores to be executed
    @param N_jobs: Number of jobs per snippet (snippet_size)
    @param N_applications: Number of applications
    @param heterogeneous_PEs: Flag to identify whether a heterogeneous SoC configuration is being used
    '''
    if heterogeneous_PEs:
        DVFS_config_list_prod = itertools.product(*DVFS_config_list)
    else:
        DVFS_config_list_prod = itertools.product(DVFS_config_list, repeat=common.num_PEs_TRACE)

    job_list = multinomial_combinations(N_jobs, N_applications)

    num_sim, config_list = merge_hardware_counters_multiple_applications(DVFS_config_list_prod, N_little_list, N_big_list, job_list)

    print("Number of job configurations:", len(job_list))
    print("Number of simulations:", num_sim)

    pool = multiprocessing.Pool(min(num_sim, multiprocessing.cpu_count()))
    # Run all simulations
    pool.map(run_sim_initial_dataset, config_list)
    pool.close()
    return 0

def merge_hardware_counters_multiple_applications(DVFS_config_list_prod, N_little_list, N_big_list, job_list):
    '''!
    Merge the hardware counters from multiple applications to create the snippet profile.
    @param DVFS_config_list_prod: List of all combinations of the available DVFS configs for all clusters
    @param N_little_list: List of all number of little cores
    @param N_big_list: List of all number of big cores
    @param job_list: Number of jobs per snippet
    @return The number of simulations and the list of configurations that will be executed
    '''
    num_sim = 0
    pool = multiprocessing.Pool(min(len(job_list)*len(N_big_list)*len(N_little_list), multiprocessing.cpu_count()))

    # First prepare and merge the different HW counter trace files into a single CSV
    merge_hardware_counters_single_app()
    hardware_counters_single_app = pd.read_csv(common.HARDWARE_COUNTERS_SINGLE_APP)

    # Create dataframe for merged counters
    merged_counters = pd.DataFrame(columns=hardware_counters_single_app.columns)
    merged_counters = merged_counters.drop(['Workload Name'], axis=1)
    merged_counters.insert(loc=0, column='Job List', value='')
    params = []
    print("Merging hardware counters...")
    for config in DVFS_config_list_prod:
        if len(config) < 2:
            print("[E] Trace generation must have at least little and big clusters, check run_parallel_sims method")
            sys.exit()
        freq_little = float(config[0].split('-')[1]) / 1000
        freq_big = float(config[1].split('-')[1]) / 1000
        for N_little in N_little_list:
            for N_big in N_big_list:
                for job_config in job_list:
                    params.append([num_sim, job_config, N_little, N_big, freq_little, freq_big, config, hardware_counters_single_app])
                    num_sim += 1
    config_list, merged_counters_list = zip(*pool.map(create_config_list_job_list, params))
    merged_counters = merged_counters.append(pd.DataFrame(merged_counters_list, columns=merged_counters.columns), ignore_index=True)
    merged_counters.to_csv(common.HARDWARE_COUNTERS_TRACE, index=False)
    print(len(config_list))
    pool.close()

    return (num_sim, config_list)

def create_dataset_IL_DTPM(timestamp, PEs):
    '''!
    Create the dataset used for the IL DTPM policies
    @param timestamp: Current timestamp
    @param PEs: List of PEs
    '''
    if (common.CREATE_DATASET_DTPM):
        if not os.path.exists(common.HARDWARE_COUNTERS_TRACE):
            print("[E] Hardware counters file was not found. This file is created by generate_traces.py and it is needed for creating the DTPM dataset.")
            print("[E] If not running an IL policy, please disable the create_dataset_DTPM flag")
            sys.exit()

        if common.job_list == []:
            print("[W] A job_list is required to create the IL dataset, please add a job_list or disable the create_dataset_DTPM flag.")
        else:
            create_header = False
            if not (os.path.exists(common.DATASET_FILE_DTPM.split(".")[0] + "__" + str(common.trace_file_num) + ".csv")):
                # Create the CSV header
                create_header = True
            with open(common.DATASET_FILE_DTPM.split(".")[0] + "__" + str(common.trace_file_num) + ".csv", 'a', newline='') as csvfile:
                dataset = csv.writer(csvfile, delimiter=',')

                if common.generate_complete_trace:
                    policy_freq_list = []
                    for cluster in common.ClusterManager.cluster_list:
                        if cluster.type != "MEM":
                            policy_freq_list.append(cluster.policy_frequency / 1000)
                    current_state = DASH_Sim_utils.get_system_state(PEs, input_freq=policy_freq_list)
                else:
                    current_state = DASH_Sim_utils.get_system_state(PEs)

                total_energy_consumption = 0
                for cluster in common.ClusterManager.cluster_list:
                    if cluster.type != "MEM":
                        current_state['Utilization_PE_' + str(cluster.ID)] = 0
                for PE in PEs:
                    cluster = common.ClusterManager.cluster_list[PE.cluster_ID]
                    if cluster.type != "MEM":
                        exec_time_samples = len(PE.utilization_list) * common.sampling_rate
                        utilization_time = 0
                        for utilization in PE.utilization_list:
                            utilization_time += common.sampling_rate * utilization
                        current_state['Utilization_PE_' + str(PE.cluster_ID)] += utilization_time / exec_time_samples
                        PE.utilization_list = []
                        total_energy_consumption += PE.snippet_energy
                for cluster in common.ClusterManager.cluster_list:
                    if cluster.type != "MEM":
                        current_state['Utilization_PE_' + str(cluster.ID)] /= len(cluster.PE_list)

                snippet_exec_time = timestamp - common.snippet_start_time

                current_state['Max_temp'] = max(common.snippet_temp_list)
                current_state['Min_temp'] = min(common.snippet_temp_list)
                current_state['Avg_temp'] = mean(common.snippet_temp_list)
                current_state['Throttling State'] = common.snippet_throttle

                current_state['Execution Time (s)'] = snippet_exec_time * 1e-9

                current_state['Energy Consumption (J)'] = total_energy_consumption

                if create_header == True:
                    header = []
                    for index, val in current_state.iteritems():
                        header.append(index)
                    dataset.writerow(header)

                dataset.writerow(current_state)

            if common.ClusterManager.cluster_list[0].DVFS == "imitation-learning":
                # Get oracle
                oracle_row = common.oracle_config_dict[str(common.current_job_list)]
                # -- Freq
                oracle_freq = []
                for cluster_ID in range(len(common.ClusterManager.cluster_list) - 1):
                    oracle_freq.append(oracle_row['FREQ_PE_' + str(cluster_ID) + ' (GHz)'])
                # -- Num_cores
                oracle_num_cores = []
                oracle_num_cores.append(oracle_row['N_little'])
                oracle_num_cores.append(oracle_row['N_big'])
                # -- Regression
                oracle_exec_time = current_state['Execution Time (s)']
                current_state_regression = copy.deepcopy(current_state)
                current_state_regression = current_state_regression.drop(['Execution Time (s)'])

                if common.aggregate_data_freq:
                    # Aggregate data
                    DTPM_policies.aggregate_data('freq', copy.deepcopy(current_state), oracle_freq)
                    common.aggregate_data_freq = False
                if common.aggregate_data_num_cores:
                    # Aggregate data
                    DTPM_policies.aggregate_data('num_cores', copy.deepcopy(current_state), oracle_num_cores)
                    common.aggregate_data_num_cores = False
                if common.aggregate_data_regression:
                    # Aggregate data
                    DTPM_policies.aggregate_data('regression', current_state_regression, oracle_exec_time)
                    common.aggregate_data_regression = False

                if common.enable_thermal_management:
                    update_temperature_dataset('freq', copy.deepcopy(current_state), oracle_freq, 'complete', PEs)
                    update_temperature_dataset('freq', copy.deepcopy(current_state), oracle_freq, 'reduced', PEs)
                    if common.enable_num_cores_prediction:
                        update_temperature_dataset('num_cores', copy.deepcopy(current_state), oracle_num_cores, 'complete', PEs)
                        update_temperature_dataset('num_cores', copy.deepcopy(current_state), oracle_num_cores, 'reduced', PEs)
                    if common.enable_real_time_constraints:
                        update_temperature_dataset('regression', current_state_regression, oracle_exec_time, 'complete', PEs)
                        update_temperature_dataset('regression', current_state_regression, oracle_exec_time, 'reduced', PEs)
                    common.thermal_limit_violated = False

def create_config_list_job_list(params):
    '''!
    Read the hardware counter profiles and compute the snippet hardware counters
    @param params: Unique parameters that identify each configuration
    @return Snippet configuration and merged counters
    '''
    num_sim, job_config, N_little, N_big, freq_little, freq_big, cfg_sample, hardware_counters_single_app = params
    total_instructions = 0
    total_cycles = 0
    total_branch = 0
    total_misses = 0
    total_mem_access = 0
    total_non_cache = 0
    for job_ID, num_job in enumerate(job_config):
        sample = hardware_counters_single_app[(hardware_counters_single_app['Workload Name'] == job_ID) &
                                              (hardware_counters_single_app['N_little'] == N_little) &
                                              (hardware_counters_single_app['N_big'] == N_big) &
                                              (hardware_counters_single_app['FREQ_PE_0 (GHz)'] == freq_little) &
                                              (hardware_counters_single_app['FREQ_PE_1 (GHz)'] == freq_big)]
        if len(sample) != 1:
            print("[E] Invalid sample detected in the hardware counter file. The query must return 1 sample.")
            print(job_ID, N_little, N_big, freq_little, freq_big)
            sys.exit()
        total_instructions += int(sample['Instructions']) * num_job
        total_cycles += int(sample['CPU Cycles']) * num_job
        total_branch += int(sample['Branch Miss Prediction']) * num_job
        total_misses += int(sample['Level 2 cache misses']) * num_job
        total_mem_access += int(sample['Data Memory Access']) * num_job
        total_non_cache += int(sample['Non-cache External Mem. Req.']) * num_job
    mt_merged_counter = {'Job List': str(list(job_config)),
                          'N_little': N_little,
                          'N_big': N_big,
                          'FREQ_PE_0 (GHz)': freq_little,
                          'FREQ_PE_1 (GHz)': freq_big,
                          'Instructions': int(total_instructions),
                          'CPU Cycles': int(total_cycles),
                          'Branch Miss Prediction': int(total_branch),
                          'Level 2 cache misses': int(total_misses),
                          'Data Memory Access': int(total_mem_access),
                          'Non-cache External Mem. Req.': int(total_non_cache)}
    mt_config = (num_sim, cfg_sample, [list(job_config)], N_little, N_big, common.num_PEs_TRACE)
    return mt_config, mt_merged_counter

def merge_hardware_counters_single_app():
    '''!
    Based on the Odroid-XU3 hardware counter profiles, filter and merge the hardware counters for individual applications into a single file.
    '''
    for i, trace_name in enumerate(common.MERGE_LIST):
        print("Trace name:", trace_name)

        trace = pd.read_csv(trace_name).drop(['Thread Number', 'Time Stamp (s)', 'A7_power (W)', 'A15_power (W)', 'Memory (W)', 'GPU (W)',
                                              'Sample Name', 'Execution Time (s)',
                                              ' TEMP_CPU_0 (C)', 'TEMP_CPU_1 (C)', 'TEMP_CPU_2 (C)', 'TEMP_CPU_3 (C)', 'TEMP_CPU_4 (C)',
                                              'Util_0(%)', 'Util_1(%)', 'Util_2(%)', 'Util_3(%)',
                                              'Util_4(%)', 'Util_5(%)', 'Util_6(%)', 'Util_7(%)',
                                              'CPU_Online_0', 'CPU_Online_1', 'CPU_Online_2', 'CPU_Online_3',
                                              'CPU_Online_4', 'CPU_Online_5', 'CPU_Online_6', 'CPU_Online_7',
                                              'FREQ_CPU_1 (GHz)', 'FREQ_CPU_2 (GHz)', 'FREQ_CPU_3 (GHz)',
                                              'FREQ_CPU_5 (GHz)', 'FREQ_CPU_6 (GHz)', 'FREQ_CPU_7 (GHz)'], axis=1).astype(float)
        trace = trace.rename(columns={'FREQ_CPU_0 (GHz)': 'FREQ_PE_0 (GHz)',
                                      'FREQ_CPU_4 (GHz)': 'FREQ_PE_1 (GHz)'})

        # Get average hardware counters across all iterations
        trace_avg = trace.groupby(['Workload Name', 'N_little', 'N_big',
                                   'FREQ_PE_0 (GHz)', 'FREQ_PE_1 (GHz)']).mean()
        hardware_counters = trace_avg.reset_index(drop=False).drop(['Iteration Num'], axis=1)

        # Adjust workload name according to the pre-defined order
        hardware_counters['Workload Name'] = i

        if i == 0:
            hardware_counters.to_csv(common.HARDWARE_COUNTERS_SINGLE_APP, mode='w', index=False, header=True)
        else:
            hardware_counters.to_csv(common.HARDWARE_COUNTERS_SINGLE_APP, mode='a', index=False, header=False)

def multinomial_combinations(n, k):
    '''!
    Return a list (2d numpy array) of all length k sequences of
    non-negative integers n1, ..., nk such that n1 + ... + nk = n.
    @param n: Number of jobs per snippet
    @param k: Number of applications
    @return List containing all combinations of n jobs and k applications that create a snippet
    '''
    bar_placements = itertools.combinations(range(1, n+k), k-1)
    tmp = [(0,) + x + (n+k,) for x in bar_placements]
    sequences =  np.diff(tmp) - 1
    return list(sequences[::-1])

def window_list(seq, n=2):
    '''!
    Return a sliding window_list (of width n) over data from the iterable, s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    '''
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def locate_min(a):
    '''!
    Return a vector with the min elements of a given list
    '''
    smallest = min(a)
    return [index for index, element in enumerate(a)
            if smallest == element]

def oracle_test_RT_and_thermal(execution_time, deadline, max_temp):
    '''!
    Test whether the current sample satifies the RT and temperature constraints
    @param execution_time: Execution time of the current snippet
    @param deadline: Execution deadline of the snippet
    @param max_temp: Temperature limit of the SoC
    @return Whether this sample satifies the constraints
    '''
    if ((common.enable_real_time_constraints and execution_time <= deadline) or common.enable_real_time_constraints is False or
                 common.real_time_aware_oracle is False):
        if (common.enable_thermal_management and max_temp < common.DTPM_thermal_limit) or common.enable_thermal_management is False:
            return True
        else:
            return False
    else:
        return False

def find_oracle_group(arg_list):
    '''!
    Return oracle decisions for a given group of snippets
    @param arg_list: Job_list and a list of samples that belong to a specific configuration
    @return The oracle decision for a given snippet
    '''
    job_list = arg_list[0]
    group = arg_list[1]

    if common.enable_real_time_constraints and common.deadline_dict == {}:
        common.deadline_dict = get_snippet_deadlines()

    min_EDP = math.inf
    min_exec_time = math.inf
    oracle_row = []
    for index, row in group.iterrows():
        execution_time = float(row['Execution Time (s)'])
        energy_consumption = float(row['Energy Consumption (J)'])
        EDP_row = energy_consumption * execution_time
        if common.enable_real_time_constraints and job_list in common.deadline_dict:
            deadline = common.deadline_dict[job_list]
        else:
            deadline = math.inf
        if (common.optimization_objective == "EDP") and (EDP_row < min_EDP):
            if oracle_test_RT_and_thermal(execution_time, deadline, row['Max_temp']):
                min_EDP = EDP_row
                oracle_row = row
        elif (common.optimization_objective == "performance") and (execution_time < min_exec_time):
            if oracle_test_RT_and_thermal(execution_time, deadline, row['Max_temp']):
                min_exec_time = execution_time
                oracle_row = row

    return job_list, oracle_row

def get_oracle_frequencies_and_num_cores():
    '''!
    Create a dictionary that stores the oracle configuration for for each group <Workload, Snippet ID>
    @return Dictionary with oracle configurations
    '''
    load_hardware_counters()

    if os.path.exists(common.DATASET_FILE_DTPM):
        dataset = pd.read_csv(common.DATASET_FILE_DTPM)
    else:
        dataset = pd.read_csv(common.DATASET_FILE_DTPM.split('.')[0] + "_freq_oracle.csv")

    # If the number of cores prediction is disabled, only consider N_big = 4 and N_little = 4 for the oracle generation
    if not common.enable_num_cores_prediction:
        dataset = dataset[(dataset['N_big'] == 4) & (dataset['N_little'] == 4)]

    grouped_tid_jid = dataset.groupby(['Job List'])
    num_groups = len(grouped_tid_jid)

    pool = multiprocessing.Pool(min(num_groups, multiprocessing.cpu_count()))
    config_list = pool.map(find_oracle_group, [[name, group] for name, group in grouped_tid_jid])
    pool.close()
    grouped_tid_jid = None

    config_dict = {}
    for list in config_list:
        sample_ID, oracle_row = list
        config_dict.update({sample_ID : oracle_row})
    return config_dict

def add_oracle_to_dataset(oracle_type):
    '''!
    Read the dataset and add the oracle decision for all samples
    @param oracle_type: Oracle type e.g., frequency, number of cores, execution time (regression)
    '''
    print("Updating dataset... ({})".format(oracle_type))
    if oracle_type == 'Frequency':
        oracle_file = common.DATASET_FILE_DTPM.split(".")[0] + "_freq_oracle.csv"
    elif oracle_type == 'Num_cores':
        oracle_file = common.DATASET_FILE_DTPM.split(".")[0] + "_num_cores_oracle.csv"
    elif oracle_type == 'Regression':
        oracle_file = common.DATASET_FILE_DTPM.split(".")[0] + "_regression_oracle.csv"
    else:
        print("[E] Error while generating the oracle dataset, type {} not recognized".format(oracle_type))
        sys.exit()
    oracle_dataset = open(oracle_file, "w")
    for i, chunk in enumerate(pd.read_csv(common.DATASET_FILE_DTPM, chunksize=1000000, iterator=True)):
        print("Loading chunk {}...".format(i))
        for index, row in pd.DataFrame(chunk).iterrows():
            oracle_row = common.oracle_config_dict[row['Job List']]
            if oracle_type == 'Frequency':
                oracle_freq = []
                for PE_ID in range(len(common.ClusterManager.cluster_list) - 1):
                    oracle_freq.append(oracle_row['FREQ_PE_' + str(PE_ID) + ' (GHz)'])
                chunk.at[index, 'Oracle_freq'] = str(oracle_freq)
            elif oracle_type == 'Num_cores':
                oracle_num_cores = []
                oracle_num_cores.append(int(oracle_row['N_little']))
                oracle_num_cores.append(int(oracle_row['N_big']))
                chunk.at[index, 'Oracle_num_cores'] = str(oracle_num_cores)
        if oracle_type == 'Regression':
            # Re-order the exec time column to the last position
            cols = list(chunk.columns.values)
            cols.remove('Execution Time (s)')
            cols.append('Execution Time (s)')
            chunk = chunk.reindex(columns=cols)
            chunk = chunk.rename(columns={'Execution Time (s)': 'Oracle_regression'})
        if i == 0:
            chunk.to_csv(oracle_dataset, mode='a', sep=',', index=False, header=True)
        else:
            chunk.to_csv(oracle_dataset, mode='a', sep=',', index=False, header=False)
    oracle_dataset.close()

def create_reduced_dataset(oracle_type):
    '''!
    Create dataset for leave-one-out experiments, completely remove one application from the dataset, configured in config_file.ini.
    @param oracle_type: Oracle type e.g., frequency, number of cores, execution time (regression)
    '''
    if oracle_type == 'Frequency':
        oracle_file = common.DATASET_FILE_DTPM.split(".")[0] + "_freq_oracle.csv"
    elif oracle_type == 'Num_cores':
        oracle_file = common.DATASET_FILE_DTPM.split(".")[0] + "_num_cores_oracle.csv"
    elif oracle_type == 'Regression':
        oracle_file = common.DATASET_FILE_DTPM.split(".")[0] + "_regression_oracle.csv"
    else:
        print("[E] Error while generating the oracle dataset, type {} not recognized".format(oracle_type))
        sys.exit()
    # Reduce dataset
    dataset = pd.read_csv(oracle_file)
    dataset_size = dataset.shape[0]
    print("Complete dataset:", dataset_size)

    num_jobs = len(dataset['Job List'][0].split(','))
    search_string = '[[]'
    for i in range(num_jobs):
        if i == 0:
            if i == common.remove_app_ID:
                search_string += '0'
            else:
                search_string += '.*'
        else:
            if i == common.remove_app_ID:
                search_string += ', 0'
            else:
                search_string += ', .*'
    search_string += '[]]'
    reduced_dataset = dataset[dataset['Job List'].str.contains(search_string, regex=True)]

    reduced_dataset_size = reduced_dataset.shape[0]
    print("Reduced dataset:", reduced_dataset_size)
    print("Fraction: {:2f}".format(reduced_dataset_size/dataset_size))

    reduced_dataset.to_csv(oracle_file.split('.')[0] + "_reduced.csv", index=False)

def update_oracle_dataset(dataset_name, dataset_type):
    '''!
    Update datasets with current oracle decisions after the DAgger iteration finishes.
    @param dataset_name: Name of the dataset
    @param dataset_type: Complete or reduced dataset
    '''
    if dataset_type == 'complete':
        dataset_file = common.DATASET_FILE_DTPM.split('.')[0] + "_" + dataset_name + "_oracle.csv"
    else:
        dataset_file = common.DATASET_FILE_DTPM.split('.')[0] + "_" + dataset_name + "_oracle_reduced.csv"
    with open(dataset_file, "r") as f:
        dataset = list(csv.reader(f))
    with open(dataset_file, "w") as f:
        writer = csv.writer(f)
        for i, row in enumerate(dataset):
            if i == 0:
                writer.writerow(row)
            else:
                oracle_row = common.oracle_config_dict[row[0]]
                if dataset_name == 'freq':
                    oracle_freq = []
                    for PE_ID in range(len(common.ClusterManager.cluster_list) - 1):
                        oracle_freq.append(oracle_row['FREQ_PE_' + str(PE_ID) + ' (GHz)'])
                    row[-1] = str(oracle_freq)
                    writer.writerow(row)
                elif dataset_name == 'num_cores':
                    oracle_num_cores = []
                    oracle_num_cores.append(int(oracle_row['N_little']))
                    oracle_num_cores.append(int(oracle_row['N_big']))
                    row[-1] = str(oracle_num_cores)
                    writer.writerow(row)
                elif dataset_name == 'regression':
                    row[-1] = str(oracle_row['Execution Time (s)'])
                    writer.writerow(row)

def sed_inplace(filename, pattern, repl):
    '''!
    Perform the pure-Python equivalent of in-place `sed` substitution: e.g.,
    `sed -i -e 's/'${pattern}'/'${repl}' "${filename}"`.
    '''
    pattern_compiled = re.compile(pattern)
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        with open(filename) as src_file:
            for line in src_file:
                tmp_file.write(pattern_compiled.sub(repl, line))
    shutil.copystat(filename, tmp_file.name)
    shutil.move(tmp_file.name, filename)

def update_temperature_dataset(dataset_name, system_state, oracle, dataset_type, PEs):
    '''!
    Update datasets with current temperature profiles
    @param dataset_name: Name of the dataset
    @param system_state: Current system state
    @param oracle: Oracle decision
    @param dataset_type: Complete or reduced dataset
    @param PEs: List of PEs
    '''
    if dataset_type == 'complete':
        dataset_file = common.DATASET_FILE_DTPM.split('.')[0] + "_" + dataset_name + "_oracle.csv"
    else:
        dataset_file = common.DATASET_FILE_DTPM.split('.')[0] + "_" + dataset_name + "_oracle_reduced.csv"
    if common.first_DAgger_iteration:
        # Predict the temperature for all snippet configurations in the first DAgger iteration
        dataset = pd.read_csv(dataset_file)
        for idx, row in dataset.iterrows():
            if str(row['Job List']) == str(system_state['Job List']):
                power_list = get_power_list_prediction(row, PEs, common.snippet_initial_temp)
                predicted_temperature = DTPM_power_models.predict_temperature_N_steps(power_list)

                predicted_temperature_value = max(predicted_temperature)
                dataset.at[idx, 'Max_temp'] = predicted_temperature_value
                dataset.at[idx, 'Min_temp'] = predicted_temperature_value
                dataset.at[idx, 'Avg_temp'] = predicted_temperature_value
        dataset.to_csv(dataset_file, mode='w', index=False, header=True)
    elif platform == "linux":
        search_string = "\"\\" + str(system_state['Job List']).split(']')[0] + "\\]\"," + str(system_state['N_little']) + "," + str(system_state['N_big']) + ","
        for cluster_ID in range(len(common.ClusterManager.cluster_list) - 1):
            search_string += str(system_state['FREQ_PE_' + str(cluster_ID) + ' (GHz)']) + ","

        if len(common.snippet_temp_list) > 0:
            system_state['Max_temp'] = max(common.snippet_temp_list)
            system_state['Min_temp'] = min(common.snippet_temp_list)
            system_state['Avg_temp'] = mean(common.snippet_temp_list)
            system_state['Throttling State'] = common.snippet_throttle
        system_state['Oracle_' + str(dataset_name)] = str(oracle)

        replace_line = ""
        for i, item in enumerate(system_state):
            if i == 0:  # First item is a list "Job list"
                replace_line += "\"\\" + str(item).split(']')[0] + "\\]\","
            elif i == len(system_state) - 1 and (dataset_name == 'freq' or dataset_name == 'num_cores'):  # Last item is a list for freq and num_cores datasets
                replace_line += "\"\\" + str(item).split(']')[0] + "\\]\""
            else:
                replace_line += str(item)
                if i != len(system_state) - 1:
                    replace_line += ","
        replace_line_command = "sed -i \'/" + search_string + "/c\\" + replace_line + "\' " + dataset_file
        os.system(replace_line_command)
    else:
        search_string = "\"\\" + str(system_state['Job List']).split(']')[0] + "\\]\"," + str(system_state['N_little']) + "," + str(system_state['N_big']) + ","
        for cluster_ID in range(len(common.ClusterManager.cluster_list) - 1):
            search_string += str(system_state['FREQ_PE_' + str(cluster_ID) + ' (GHz)']) + ","
        search_string +=".*"

        if len(common.snippet_temp_list) > 0:
            system_state['Max_temp'] = max(common.snippet_temp_list)
            system_state['Min_temp'] = min(common.snippet_temp_list)
            system_state['Avg_temp'] = mean(common.snippet_temp_list)
            system_state['Throttling State'] = common.snippet_throttle
        system_state['Oracle_' + str(dataset_name)] = str(oracle)

        replace_line = ""
        for i, item in enumerate(system_state):
            if i == 0:  # First item is a list "Job list"
                replace_line += "\"" + str(item).split(']')[0] + "]\","
            elif i == len(system_state) - 1 and (dataset_name == 'freq' or dataset_name == 'num_cores'):  # Last item is a list for freq and num_cores datasets
                replace_line += "\"" + str(item).split(']')[0] + "]\""
            else:
                replace_line += str(item)
                if i != len(system_state) - 1:
                    replace_line += ","
        sed_inplace(dataset_file, search_string, replace_line)

def get_power_list_prediction(row, PEs, input_temperature):
    '''!
    Predict the power for all clusters based on the inputs
    @param row: Dataset row (sample)
    @param PEs: List of PEs
    @param input_temperature: Baseline temperature
    @return Power consumption prediction for all clusters
    '''
    power_list = []
    for cluster in common.ClusterManager.cluster_list:
        if cluster.type != 'MEM':
            N_cores = 1
            if cluster.type == 'LTL':
                N_cores = int(row['N_little'])
            elif cluster.type == 'BIG':
                N_cores = int(row['N_big'])

            num_tasks = int(mean(cluster.snippet_num_tasks_list))
            # num_tasks = int(max(cluster.snippet_num_tasks_list))
            num_tasks = max(min(num_tasks, N_cores), 1)

            frequency = int(float(row['FREQ_PE_' + str(cluster.ID) + ' (GHz)']) * 1000)
            voltage = DTPM_power_models.get_voltage_constant_mode(cluster.OPP, frequency)
            static_power = DTPM_power_models.compute_static_power_dissipation(cluster.ID, input_temperature=input_temperature,
                                                                              input_voltage=voltage)

            max_power_consumption, freq_threshold = DTPM_power_models.get_max_power_consumption(cluster, PEs, N_tasks=num_tasks,
                                                                                                N_cores=N_cores)
            max_power_core = max_power_consumption / num_tasks
            Cdyn_alpha = DTPM_power_models.compute_Cdyn_and_alpha(None, max_power_core, freq_threshold, OPP=cluster.OPP)
            dynamic_power = DTPM_power_models.compute_dynamic_power_dissipation(frequency, voltage, Cdyn_alpha)

            total_power = static_power * N_cores + dynamic_power * min(num_tasks, N_cores)
            power_list.append(total_power)
    return power_list

def get_snippet_deadlines():
    '''!
    Get the deadlines for all snippets
    @return Dictionary with the snippets' deadlines
    '''
    deadlines = pd.read_csv(common.DEADLINE_FILE)
    deadline_dict = {}
    for index, row in deadlines.iterrows():
        sample_ID   = row['Job List']
        deadline    = row['Deadline']
        deadline_dict.update({sample_ID: deadline})
    return deadline_dict

def load_hardware_counters():
    '''!
    Update the global hardware counter variable with the trace for the hardware counters, normalizing the counters w.r.t. the number of instructions.
    '''
    if os.path.exists(common.HARDWARE_COUNTERS_TRACE):
        common.hardware_counters = pd.read_csv(common.HARDWARE_COUNTERS_TRACE)

        # Normalize hardware counters w.r.t. the number of instructions
        common.hardware_counters['CPU Cycles'] = common.hardware_counters['CPU Cycles'] / common.hardware_counters['Instructions']
        common.hardware_counters['Branch Miss Prediction'] = common.hardware_counters['Branch Miss Prediction'] / common.hardware_counters['Instructions']
        common.hardware_counters['Level 2 cache misses'] = common.hardware_counters['Level 2 cache misses'] / common.hardware_counters['Instructions']
        common.hardware_counters['Data Memory Access'] = common.hardware_counters['Data Memory Access'] / common.hardware_counters['Instructions']
        common.hardware_counters['Non-cache External Mem. Req.'] = common.hardware_counters['Non-cache External Mem. Req.'] / common.hardware_counters['Instructions']

        common.hardware_counters = common.hardware_counters.drop(['Instructions'], axis=1)