'''!
@brief This file is the main() function which should be run to get the simulation results.
'''
import simpy
import configparser
import matplotlib.pyplot as plt                                                 
import random                                                                  
import numpy as np
import sys
import os
import networkx as nx
import pickle
import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


import job_generator                                                            # Dynamic job generation is handled by job_generator.py
import common                                                                   # The common parameters used in DASH-Sim are defined in common.py
import DASH_SoC_parser                                                          # The resource parameters used in DASH-Sim are obtained from resource initialization file(SoC.**.txt), parsed by DASH_SoC_parser.py
import job_parser                                                               # The parameters in a job used in DASH-Sim are obtained from Job initialization file (job_**.txt), parsed by job_parser.py
import processing_element                                                       # Define the processing element class
import DASH_Sim_core                                                            # The core of the simulation engine (SimulationManager) is defined DASH_Sim_core.py
import scheduler                                                                # The DASH-Sim uses the scheduler defined in scheduler.py
import DASH_Sim_utils
import DTPM_utils
import ils_scripts
import DASH_acc_parser

def run_simulator(scale_values=common.scale_values_list):
    '''!
    Parse the job and SoC configurations and execute the simulation environment with the parameters from config_file.ini
    @param scale_values: Optional input to select specific scale values. Default value is defined in the config_file.ini
    '''

    #common.clear_screen()                                                           # Clear IPthon Console screen at the beginning of each simulation
    print('%59s'%('**** Welcome to DASH_Sim.v0 ****'))
    print('%65s'%('**** \xa9 2020 eLab ASU ALL RIGHTS RESERVED ****'))

    # Instantiate the ResourceManager object that contains all the resources
    # in the target DSSoC
    resource_matrix = common.ResourceManager()                                      # This line generates an empty resource matrixesource_matrix = common.ResourceManager()                                      # This line generates an empty resource matrix
    common.resource_matrix_Acc = common.ResourceManagerAcc()
    config = configparser.ConfigParser()
    config.read('config_file.ini')
    resource_file = "config_SoC/" + config['DEFAULT']['resource_file']
    DASH_SoC_parser.resource_parse(resource_matrix, resource_file)    # Parse the input configuration file to populate the resource matrix
    DASH_acc_parser.acc_parse()

    ## Open report file handles
    common.ils_enable_dataset_save     = config.getboolean('IL SCHEDULER', 'enable_dataset_save')
    common.ils_enable_policy_decision  = config.getboolean('IL SCHEDULER', 'enable_ils_policy')
    common.ils_enable_dagger           = config.getboolean('IL SCHEDULER', 'enable_ils_dagger')
    common.scheduler = config['DEFAULT']['scheduler']  # Assign scheduler name variable
    config_scale_values = config['SIMULATION MODE']['scale_values']
    common.scale_values_list = common.str_to_list(config_scale_values)
    os.makedirs('./reports', exist_ok=True)
    os.makedirs('./datasets', exist_ok=True)
    if common.ils_enable_dataset_save or common.ils_enable_policy_decision or common.ils_enable_dagger:
        os.makedirs('./reports', exist_ok=True)
        ils_scripts.open_report_file_handles(common.scheduler, scale_values)
        common.scale_values_list = common.str_to_list(scale_values)
    plt.close('all')                                                                # close all existing plots before the new simulation

    # Print file headers for DAS scheduler if dataset is to be stored
    if common.scheduler == 'DAS':
        common.scale_values_list = common.str_to_list(scale_values)
        if common.das_dataset:
            os.makedirs('./datasets/' + common.das_dataset_folder, exist_ok=True)
            das_filename = './datasets/' + common.das_dataset_folder + '/data_DAS_'         
            for i in range(len(common.scale_values_list)):
                das_filename += str(common.scale_values_list[i])
            for prob in common.job_probabilities:
                das_filename += '_' + str(prob)
            das_filename += '.csv'
            ils_scripts.das_print_file_headers(das_filename, common.ClusterManager)
    ## if common.scheduler == 'DAS':
    if (common.CLEAN_TRACES):
        DASH_Sim_utils.clean_traces()

    for cluster in common.ClusterManager.cluster_list:
        if cluster.DVFS != 'none':
            if len(cluster.trip_freq) != len(common.trip_temperature) or len(cluster.trip_freq) != len(common.trip_hysteresis):
                print("[E] The trip points must match in size:")
                print("[E] Trip frequency (SoC file):      {} (Cluster {})".format(len(cluster.trip_freq), cluster.ID))
                print("[E] Trip temperature (config file): {}".format(len(common.trip_temperature)))
                print("[E] Trip hysteresis (config file):  {}".format(len(common.trip_hysteresis)))
                sys.exit()
            if len(cluster.power_profile) != len(cluster.PG_profile):
                print("[E] The power and PG profiles must match in size, please check the SoC file")
                print("[E] Cluster ID: {}, Num power points: {}, PG power points: {}".format(cluster.ID, len(cluster.power_profile), len(cluster.PG_profile)))
                sys.exit()

    ## Report file handles
    if common.simulation_reports :
        os.makedirs('./reports', exist_ok=True)
        if common.scheduler == 'DAS':
            if common.das_complex_only:
                report_filename = './reports/report_ETF_' + common.scheduler
            elif common.das_policy:
                report_filename = './reports/report_policy_' + common.scheduler                
            else:
                report_filename = './reports/report_dataset_' + common.scheduler
        else:
            report_filename = './reports/report_' + common.scheduler
        for scale in common.scale_values_list:
            report_filename += '_' + str(scale) 
        report_filename += '.rpt'
        common.report_fp = open(report_filename, 'w')
    
    # Instantiate the ApplicationManager object that contains all the jobs
    # in the target DSSoC
    jobs = common.ApplicationManager()                                              # This line generates an empty list for all jobs
    job_files_list = ["config_Jobs/" + f for f in common.str_to_list(config['DEFAULT']['job_file'])]
    for job_file in job_files_list:
        job_parser.job_parse(jobs, job_file)                                        # Parse the input job file to populate the job list

    ## Initialize variables at simulation start
    DASH_Sim_utils.init_variables_at_sim_start()

    if common.job_list == []:
        if len(common.job_probabilities) != len(job_files_list):
            print("[E] The length of the application list (job_file) must match the job_probabilities configuration.")
            print("[E] Please check these parameters in the config_file.ini")
            sys.exit()
    else:
        if len(common.job_list[0]) != len(job_files_list):
            print("[E] The length of the application list (job_file) must match each snippet in the job_list configuration.")
            print("[E] Please check these parameters in the config_file.ini")
            sys.exit()
    
    if not common.enable_real_time_constraints and common.enable_regression_policy:
        print("[E] The regression policy flag requires that the real-time constraint flag is enabled.")
        print("[E] Please enable the RT constraint flag, or disable the regression policy.")
        sys.exit()

    if common.enable_real_time_constraints:
        common.deadline_dict = DTPM_utils.get_snippet_deadlines()

    # Get the oracle configurations if using IL policy, otherwise, just load the hardware counter trace
    if common.ClusterManager.cluster_list[0].DVFS == "imitation-learning":
        common.oracle_config_dict = DTPM_utils.get_oracle_frequencies_and_num_cores()
    else:
        DTPM_utils.load_hardware_counters()

    # DTPM
    if os.path.exists(common.DTPM_freq_policy_file):
        common.IL_freq_policy = pickle.load(open(common.DTPM_freq_policy_file, 'rb'))
    else:
        common.IL_freq_policy = None

    if os.path.exists(common.DTPM_num_cores_policy_file):
        common.IL_num_cores_policy = pickle.load(open(common.DTPM_num_cores_policy_file, 'rb'))
    else:
        common.IL_num_cores_policy = None

    if os.path.exists(common.DTPM_regression_policy_file):
        common.IL_regression_policy = pickle.load(open(common.DTPM_regression_policy_file, 'rb'))
    else:
        common.IL_regression_policy = None

    # Check whether the resource_matrix and task list are initialized correctly
    if (common.DEBUG_CONFIG):
        print('\n[D] Starting DASH-Sim in DEBUG Mode ...')
        print("[D] Read the resource_matrix and write its contents")
        num_of_resources = len(resource_matrix.list)
        num_of_jobs = len(jobs.list)

        for i in range(num_of_resources):
            curr_resource = resource_matrix.list[i]
            print("[D] Adding a new resource: Type: %s, Name: %s, ID: %d, Capacity: %d" 
                  %(curr_resource.type, curr_resource.name, int(curr_resource.ID), int(curr_resource.capacity))) 
            print("[D] It supports the following %d functionalities"
                  %(curr_resource.num_of_functionalities))
    
            for ii in range(curr_resource.num_of_functionalities):
                print ('%4s'%('')+curr_resource.supported_functionalities[ii],
                       curr_resource.performance[ii])
        print('\nCommunication Bandwidth matrix between Resources is\n', common.ResourceManager.comm_band)
            # end for ii
        # end for i

        print("\n[D] Read each application and write its components")
        for ii in range(num_of_jobs):
            curr_job = jobs.list[ii]
            num_of_tasks = len(curr_job.task_list)
            print('\n%10s'%('')+'Now reading application %s' %(ii+1))
            print('Application name: %s, Number of tasks in the application: %s'%(curr_job.name, num_of_tasks))
            
            for task in jobs.list[ii].task_list:
                print("Task name: %s, Task ID: %s, Task Predecessor(s) %s"
                      %(task.name, task.ID, task.predecessors))
            print('Communication Volume matrix between Tasks is\n', jobs.list[ii].comm_vol)
        print(' ')
        # end for ii

        print('[D] Read the scheduler name')
        print('Scheduler name: %s' % common.scheduler)
        print('')
    # end if (DEBUG)


    if (common.simulation_mode == 'validation'):
        '''
        Start the simulation in VALIDATION MODE
        '''
        job_execution_time = 0                                                  # Average execution time
        
        # Provide the value of the seed for the random variables
        random.seed(common.seed)  # user can regenerate the same results by assigning a value to $random_seed in configuration file
        np.random.seed(common.seed)
        common.iteration = 1 # set the iteration value

        # Instantiate the PerfStatics object that contains all the performance statics
        common.results = common.PerfStatics()

        # Set up the Python Simulation (simpy) environment
        env = simpy.Environment(initial_time=0)
        sim_done = env.event()

        # Construct the processing elements in the target DSSoC
        DASH_resources = []

        for i,resource in enumerate(resource_matrix.list):
            # Define the PEs (resources) in simpy environment
            new_PE = processing_element.PE(env, resource.type, resource.name,
                                           resource.ID, resource.cluster_ID, resource.capacity) # Generate a new PE with this generic process
            DASH_resources.append(new_PE)
        # end for

        # Construct the scheduler
        DASH_scheduler = scheduler.Scheduler(env, resource_matrix, common.scheduler,
                                             DASH_resources, jobs)

        # Check whether PEs are initialized correctly
        if (common.DEBUG_CONFIG):
            print('[D] There are %d simpy resources.' % len(DASH_resources))
            print('[D] Completed building and debugging the DASH model.\n')


        # Start the simulation engine
        print('[I] Starting the simulation under VALIDATION MODE...')

        job_gen = job_generator.JobGenerator(env, resource_matrix, jobs, DASH_scheduler, DASH_resources)

        sim_core = DASH_Sim_core.SimulationManager(env, sim_done, job_gen, DASH_scheduler, DASH_resources,
                                                  jobs, resource_matrix)


        env.run(until = common.simulation_length)

        
        job_execution_time += common.results.cumulative_exe_time / common.results.completed_jobs                           # find the mean job duration

        print('[I] Completed Simulation ...')
        for job in common.Validation.generated_jobs:
            if job in common.Validation.completed_jobs:
                continue
            else:
              print('[E] Not all generated jobs are completed')
              sys.exit()
        print('[I] And, simulation is validated, successfully.')
        print('\nSimulation Parameters')
        print("-"*55)
        print("%-30s : %-20s"%("SoC config file",resource_file))
        print("%-30s : %-20s"%("Job config files",' '.join(job_files_list)))
        print("%-30s : %-20s"%("Scheduler",common.scheduler))
        print("%-30s : %-20s"%("Clock period(us)",common.simulation_clk))
        print("%-30s : %-20d"%("Simulation length(us)",common.simulation_length))
        print('\nSimulation Statitics')
        print("-"*55)
        print("%-30s : %-20s" % ("Execution time(us)", round(common.results.execution_time, 2)))
        print("%-30s : %-20s" % ("Cumulative Execution time(us)", round(common.results.cumulative_exe_time, 2)))
        print("%-30s : %-20s"%("Avg execution time(us)",job_execution_time))
        print("%-30s : %-20s" % ("Total energy consumption(uJ)",
                                 round(common.results.energy_consumption, 2)))
        print("%-30s : %-20s" % ("EDP",
                                 round(common.results.execution_time * common.results.energy_consumption, 2)))
        DASH_Sim_utils.trace_system()
        # End of simpy simulation

        plot_gantt_chart = True
        if plot_gantt_chart:
            # Creating a text based Gantt chart to visualize the simulation
            job_ID = -1
            ilen = len(resource_matrix.list) - 1  # since the last PE is the memory
            pos = np.arange(0.5, ilen * 0.5 + 0.5, 0.5)
            fig = plt.figure(figsize=(10, 6))
            # fig = plt.figure(figsize=(10,3.5))
            ax = fig.add_subplot(111)
            color_choices = ['red', 'blue', 'green', 'cyan', 'magenta']
            for i in range(len(resource_matrix.list)):
                for ii, task in enumerate(common.TaskQueues.completed.list):
                    if (i == task.PE_ID):
                        end_time = task.finish_time
                        start_time = task.start_time
                        ax.barh((i * 0.5) + 0.5, end_time - start_time, left=start_time,
                                height=0.3, align='center', edgecolor='black', color='white', alpha=0.95)
                        # Retrieve the job ID which the current task belongs to
                        for iii, job in enumerate(jobs.list):
                            if (job.name == task.jobname):
                                job_ID = iii
                        ax.text(0.5 * (start_time + end_time - len(str(task.ID)) - 0.25), (i * 0.5) + 0.5 - 0.03125,
                                task.ID, color=color_choices[(task.jobID) % 5], fontweight='bold', fontsize=18, alpha=0.75)
            # color_choices[(task.jobID)% 5]
            # color_choices[job_ID]
            # locsy, labelsy = plt.yticks(pos, ['P0','P1','P2']) #
            locsy, labelsy = plt.yticks(pos, range(len(resource_matrix.list)))
            plt.ylabel('Processing Element', fontsize=18)
            plt.xlabel('Time', fontsize=18)
            plt.tick_params(labelsize=16)
            # plt.title('DASH-Sim - %s' %(common.scheduler), fontsize =18)
            plt.setp(labelsy, fontsize=18)
            ax.set_ylim(bottom=-0.1, top=ilen * 0.5 + 0.5)
            ax.set_xlim(left=-5)
            ax.grid(color='g', linestyle=':', alpha=0.5)
            plt.show()
            
        
    # end of if (common.simulation_mode == 'validation'):

    if (common.simulation_mode == 'performance'):
        '''
        Start the simulation in PERFORMANCE MODE
        '''
        ave_job_injection_rate = [0]*len(common.scale_values_list)                  # The list contains the mean of the lambda injection value corresponding each lambda value 
                                                                                    # Based on the number of jobs put into ready queue list
        ave_job_execution_time = [0]*len(common.scale_values_list)                  # The list contains the mean job duration for each lambda value
        ave_throughput         = [0]*len(common.scale_values_list)                  # The list contains the total bits processed
        ave_job_completion_rate = [0]*len(common.scale_values_list)                 # The list contains the mean job completion rate for each lambda value
        lamd_values_list = [0]*len(common.scale_values_list)                        # The list of lambda values which will determine the job arrival rate
        ave_concurrent_jobs = [0]*len(common.scale_values_list)                     # Average number of jobs in the system for a workload with a specific scale value
        ave_active_time  = [0]*len(common.scale_values_list)                        # The list of average active times of PEs for a workload with a specific scale
        ave_blocking_time = [0]*len(common.scale_values_list)                       # The list of blocking times of PEs for a workload with a specific scale
        ave_energy = [0]*len(common.scale_values_list)                              # The list contains the average energy consumption for each lambda (scale) value
        ave_EDP = [0]*len(common.scale_values_list)                                 # The list contains the average EDP for each lambda value
        
        
        for (ind,scale) in enumerate(common.scale_values_list):
            common.scale = scale  # Assign each value in $scale_values_list to common.scale
            lamd_values_list[ind] = 1 / scale

            if (common.INFO_JOB):
                print('%10s'%('')+'[I] Simulation starts for scale value %s' %(scale))

            # Iterate over a fixed number of iterations
            job_execution_time  = 0.0
            job_injection_rate  = 0.0
            total_throughput    = 0.0
            job_completion_rate = 0.0
            concurrent_jobs     = 0.0
            active_time         = [0]*len(resource_matrix.list)
            blocking_time       = [0]*len(resource_matrix.list)
            energy              = 0.0
            EDP                 = 0.0

            for iteration in range(common.num_of_iterations):                       # Repeat the simulation for a given number of numbers for each lambda value
                
                ## Initialize variables at simulation start
                DASH_Sim_utils.init_variables_at_sim_start()

                ## Set a global iteration variable
                common.iteration = iteration

                random.seed(iteration)                                              # user can regenerate the same results by assigning a value to $random_seed in configuration file
                np.random.seed(iteration)

                # Instantiate the PerfStatics object that contains all the performance statics
                common.results = common.PerfStatics()
                common.computation_dict = {}
                common.current_dag = nx.DiGraph()
                if common.warmup_based_on_jobs:
                    common.warmup_period = np.inf
                # Set up the Python Simulation (simpy) environment
                env = simpy.Environment(initial_time=0)
                sim_done = env.event()

                # Construct the processing elements in the target DSSoC
                DASH_resources = []
                for i,resource in enumerate(resource_matrix.list):
                    # Define the PEs (resources) in simpy environment
                    new_PE = processing_element.PE(env, resource.type, resource.name,
                                                   resource.ID, resource.cluster_ID, resource.capacity) # Generate a new PE with this generic process
                    DASH_resources.append(new_PE)
                # end for

                # Construct the scheduler
                DASH_scheduler = scheduler.Scheduler(env, resource_matrix, common.scheduler,
                                                     DASH_resources, jobs)

                if (common.INFO_JOB):
                    print('[I] Starting iteration: %d' %(iteration+1))

                job_gen = job_generator.JobGenerator(env, resource_matrix, jobs, DASH_scheduler, DASH_resources)

                sim_core = DASH_Sim_core.SimulationManager(env, sim_done, job_gen, DASH_scheduler, DASH_resources,
                                                           jobs, resource_matrix)

                if common.inject_fixed_num_jobs is False:
                    env.run(until = common.simulation_length)
                else:
                    env.run(until = sim_done)

                # Now, the simulation has completed
                # Next, process the results
                if (common.INFO_JOB):
                    print('[I] Completed iteration: %d' %(iteration+1))
                    print('[I] Number of injected jobs: %d' %(common.results.injected_jobs))
                    print('[I] Number of completed jobs: %d' %(common.results.completed_jobs))
                    try:
                        print('[I] Ave latency: %f'
                        %(common.results.cumulative_exe_time/common.results.completed_jobs))
                    except ZeroDivisionError:
                        print('[I] No completed jobs')
                    print("[I] %-30s : %-20s" % ("Execution time(ns)", round(common.results.execution_time - common.warmup_period, 2)))
                    print("[I] %-30s : %-20s" % ("Throughput (Mbps)",  round(common.results.bits_processed * 1000 / (common.results.execution_time - common.warmup_period), 2)))
                    print("[I] %-30s : %-20s" % ("Cumulative Execution time(ns)", round(common.results.cumulative_exe_time, 2)))
                    print("[I] %-30s : %-20s" % ("Total energy consumption(J)",
                                                 round(common.results.cumulative_energy_consumption, 6)))
                    print("[I] %-30s : %-20s" % ("EDP",
                                                 round((common.results.execution_time - common.warmup_period) * common.results.cumulative_energy_consumption, 2)))
                    print("[I] %-30s : %-20s" % ("Average concurrent jobs", round(common.results.average_job_number, 2)))
                    
                    result_exec_time = common.results.execution_time - common.warmup_period
                    result_energy_cons = common.results.cumulative_energy_consumption
                    result_EDP = result_exec_time * result_energy_cons
                    header_list = ['Execution time(us)', 'Total energy consumption(J)', 'EDP']
                    result_list = [result_exec_time, result_energy_cons, result_EDP]
                    if common.total_predictions > 0:
                        print("%-30s : %-20s" % ("Total predictions",
                                                 common.total_predictions))
                        print("%-30s : %-20s" % ("Wrong predictions (freq)",
                                                 common.wrong_predictions_freq))
                        header_list.append('Total predictions')
                        result_list.append(common.total_predictions)
                        header_list.append('Wrong predictions (freq)')
                        result_list.append(common.wrong_predictions_freq)
                        if common.enable_num_cores_prediction:
                            print("%-30s : %-20s" % ("Wrong predictions (num_cores)",
                                                     common.wrong_predictions_num_cores))
                            header_list.append('Wrong predictions (num_cores)')
                            result_list.append(common.wrong_predictions_num_cores)
                        print("%-30s : %-20s" % ("Accuracy (freq)",
                                                 ((common.total_predictions - common.wrong_predictions_freq) / common.total_predictions) * 100))
                        header_list.append('Accuracy (freq)')
                        result_list.append(((common.total_predictions - common.wrong_predictions_freq) / common.total_predictions) * 100)
                        if common.enable_num_cores_prediction:
                            print("%-30s : %-20s" % ("Accuracy (num_cores)",
                                                     ((common.total_predictions - common.wrong_predictions_num_cores) / common.total_predictions) * 100))
                            header_list.append('Accuracy (num_cores)')
                            result_list.append(((common.total_predictions - common.wrong_predictions_num_cores) / common.total_predictions) * 100)
                    if common.enable_real_time_constraints and common.snippet_ID_exec > 0:
                        print("%-30s : %-20s" % ("Missed deadlines:",
                                                 "{} ({:.2f}%)".format(common.missed_deadlines, (common.missed_deadlines / (common.snippet_ID_exec)) * 100)))
                        header_list.append('Missed deadlines')
                        result_list.append(common.missed_deadlines)
                        header_list.append('Missed deadlines (%)')
                        result_list.append((common.missed_deadlines / (common.snippet_ID_exec)) * 100)
                    DASH_Sim_utils.trace_system()
                    if not common.generate_complete_trace:
                        if not os.path.exists(common.RESULTS):
                            with open(common.RESULTS, 'w', newline='') as csvfile:
                                result_file = csv.writer(csvfile, delimiter=',')
                                result_file.writerow(header_list)
                        with open(common.RESULTS, 'a', newline='') as csvfile:
                            result_file = csv.writer(csvfile, delimiter=',')
                            result_file.writerow(result_list)
                        if common.ClusterManager.cluster_list[0].DVFS == "imitation-learning" and common.enable_thermal_management:
                            common.oracle_config_dict = DTPM_utils.get_oracle_frequencies_and_num_cores()
                            DTPM_utils.update_oracle_dataset('freq', 'complete')
                            DTPM_utils.update_oracle_dataset('freq', 'reduced')
                            if common.enable_num_cores_prediction:
                                DTPM_utils.update_oracle_dataset('num_cores', 'complete')
                                DTPM_utils.update_oracle_dataset('num_cores', 'reduced')
                            if common.enable_real_time_constraints:
                                DTPM_utils.update_oracle_dataset('regression', 'complete')
                                DTPM_utils.update_oracle_dataset('regression', 'reduced')

                try:
                    job_execution_time += common.results.cumulative_exe_time / common.results.completed_jobs                    # find the mean job duration value for this iteration
                except ZeroDivisionError:
                    job_execution_time += 0

                if common.simulation_reports :
                    common.report_fp.write('[I] Completed iteration: %d\n' %(iteration+1))
                    common.report_fp.write('[I] Number of injected jobs: %d\n' %(common.results.injected_jobs))
                    common.report_fp.write('[I] Number of completed jobs: %d\n' %(common.results.completed_jobs))
                    try:
                        common.report_fp.write('[I] Ave latency: %f\n'
                        %(common.results.cumulative_exe_time/common.results.completed_jobs))
                    except ZeroDivisionError:
                        common.report_fp.write('[I] No completed jobs\n')
                    common.report_fp.write("[I] %-30s : %-20s\n" % ("Average concurrent jobs", round(common.results.average_job_number, 2)))
                    common.report_fp.write("[I] %-30s : %-20s\n" % ("Execution time(us)", round(common.results.execution_time - common.warmup_period, 2)))
                    common.report_fp.write("[I] %-30s : %-20s\n" % ("Throughput (Mbps)",  round(common.results.bits_processed * 1000 / (common.results.cumulative_exe_time - common.warmup_period), 2)))
                    common.report_fp.write("[I] %-30s : %-20s\n" % ("Cumulative Execution time(us)", round(common.results.cumulative_exe_time, 2)))

                # Add the results obtained for this iteration into a list
                job_injection_rate += common.results.injected_jobs / (common.results.execution_time - common.warmup_period)      
                total_throughput   += common.results.bits_processed * 1000 / (common.results.execution_time - common.warmup_period)      
                job_completion_rate += common.results.completed_jobs / (common.results.execution_time - common.warmup_period)    
                concurrent_jobs += common.results.average_job_number
                for i, resource in enumerate(DASH_resources):
                    active_time[i] += resource.active/common.results.execution_time
                    blocking_time[i] += resource.blocking/common.results.execution_time
                energy += common.results.cumulative_energy_consumption
                EDP += (common.results.execution_time - common.warmup_period) * common.results.cumulative_energy_consumption
            # end of for iteration in range(common.num_of_iterations):

            # Calculate average values of the results from all iterations
            ave_job_execution_time[ind] = job_execution_time / common.num_of_iterations
            ave_throughput[ind] = total_throughput / common.num_of_iterations
            ave_job_injection_rate[ind] = job_injection_rate / common.num_of_iterations
            ave_job_completion_rate[ind] = job_completion_rate / common.num_of_iterations
            ave_concurrent_jobs[ind] = concurrent_jobs / common.num_of_iterations
            ave_active_time[ind] = [x / common.num_of_iterations for x in active_time]
            ave_blocking_time[ind] = [x / common.num_of_iterations for x in blocking_time]
            ave_energy[ind] = energy / common.num_of_iterations
            ave_EDP[ind] = EDP / common.num_of_iterations
            

            if (common.INFO_JOB):
                print('[I] Completed all %d iterations for scale = %d,'
                      %(common.num_of_iterations,scale), end='')
                print(' injection rate:%f, completion rate:%f, ave_execution_time:%f, ave_throughput:%f, EDP:%f, energy:%f'
                      % (ave_job_injection_rate[ind]*1000000, ave_job_completion_rate[ind], ave_job_execution_time[ind], ave_throughput[ind],ave_EDP[ind],ave_energy[ind]))
            if common.simulation_reports :
                common.report_fp.write('[I] Completed all %d iterations for scale = %d,'
                      %(common.num_of_iterations,scale))
                common.report_fp.write(' injection rate:%f, concurrent_jobs:%f, completion rate:%f, ave_execution_time:%f, ave_throughput:%f, EDP:%f, energy:%f\n'
                      % (ave_job_injection_rate[ind]*1000000, ave_concurrent_jobs[ind], ave_job_completion_rate[ind], ave_job_execution_time[ind]/1000, ave_throughput[ind],ave_EDP[ind],ave_energy[ind]))
        
        if common.simulation_reports :
            common.report_fp.close()
        # end of for (ind,scale) in enumerate(common.scale_values_list):

if __name__ == '__main__':
    run_simulator(common.config_scale_values)
