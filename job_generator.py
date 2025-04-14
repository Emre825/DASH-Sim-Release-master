'''!
@brief This file contains the code for the job generator.
'''
import random 
import copy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
import simpy

from heft import heft, dag_merge, gantt
from peft import peft
import common
import DASH_Sim_utils
#import CP_models
           
class JobGenerator:
    '''!
    Define the JobGenerator class to handle dynamic job generation
    '''
    def __init__(self, env, resource_matrix, jobs, scheduler, PE_list):
        '''!
        @param env: Pointer to the current simulation environment
        @param resource_matrix: The data structure that defines power/performance characteristics of the PEs for each supported task
        @param jobs: The list of all jobs given to DASH-Sim
        @param scheduler: Pointer to the DASH_scheduler
        @param PE_list: The PEs available in the current SoCs
        '''
        self.env = env
        self.resource_matrix = resource_matrix
        self.jobs = jobs
        self.scheduler = scheduler
        self.PEs = PE_list
        
        
        # Initially none of the tasks are outstanding
        common.TaskQueues.outstanding = common.TaskManager()                    # List of *all* tasks waiting to be processed

        # Initially none of the tasks are completed
        common.TaskQueues.completed = common.TaskManager()                      # List of completed tasks

        # Initially none of the tasks are running on the PEs
        common.TaskQueues.running = common.TaskManager()                        # List of currently running tasks
        
        # Initially none of the tasks are completed
        common.TaskQueues.ready = common.TaskManager()                          # List of tasks that are ready for processing
        
        # Initially none of the tasks are in wait ready queue
        common.TaskQueues.wait_ready = common.TaskManager()                     # List of tasks that are waiting for being ready for processing
        
        # Initially none of the tasks are executable
        common.TaskQueues.executable = common.TaskManager()                     # List of tasks that are ready for execution
        
        self.generate_job = True                                                # Initially $generate_job is True so that as soon as run function is called
                                                                                #   it will start generating jobs
        self.max_num_jobs = common.max_num_jobs                                 # Number of jobs to be created
        self.generated_job_list = []                                            # List of all jobs that are generated
        self.offset = 0                                                         # This value will be used to assign correct ID numbers for incoming tasks

        self.action = env.process(self.run())                                   # Starts the run() method as a SimPy process

    def run(self):
        '''
        Inject new jobs in the system based on the defined injection rate (scale value in config_file.ini) and total number of jobs to be injected (inject_fixed_num_jobs and max_jobs in config_file.ini).
        '''
        i = 0  # Initialize the iteration variable
        num_jobs = 0
        count = 0
        summation = 0
        np.random.seed(common.iteration)

        
        if len(DASH_Sim_utils.get_current_job_list()) != len(self.jobs.list) and DASH_Sim_utils.get_current_job_list() != []:
            print('[E] Time %s: Job_list and job_file configs have different lengths, please check SoC.**.txt file'
                  % (self.env.now))
            sys.exit()

        while (self.generate_job):  # Continue generating jobs till #generate_job is False

            if (common.results.job_counter >= common.max_jobs_in_parallel or (common.job_list != [] and common.snippet_ID_inj == common.snippet_ID_exec)):
                try:
                    yield self.env.timeout(common.simulation_clk)
                except simpy.exceptions.Interrupt:
                    pass
            else:
                valid_jobs = []
                common.current_job_list = DASH_Sim_utils.get_current_job_list()
                for index, job_counter in enumerate(common.job_counter_list):
                    if job_counter < common.current_job_list[index]:
                        valid_jobs.append(index)
                
                if valid_jobs != []:
                    selection = np.random.choice(valid_jobs)
                    #print('selected job id is',selection)
                else:
                    num_of_apps = len(self.jobs.list)
                    selection = np.random.choice(list(range(num_of_apps)), 1, p=common.job_probabilities)
                    # print('selected job id is',selection)

                self.generated_job_list.append(copy.deepcopy(self.jobs.list[int(selection)]))               # Create each job as a deep copy of the job chosen from job list
                common.results.job_counter += 1
                summation += common.results.job_counter
                count += 1
                common.results.average_job_number = summation/count
                
                if (common.DEBUG_JOB):
                    print('[D] Time %d: Job generator added job %d' % (self.env.now, i + 1))

                if (common.simulation_mode == 'validation'):
                    common.Validation.generated_jobs.append(i)

                # Should this move in a "if (common.scheduler_type == DAG scheduler)" direction?
                if (self.scheduler.name == 'HEFT' or self.scheduler.name == 'PEFT'):
                    # Load the graph associated with this job
                    job_dag = nx.DiGraph(self.generated_job_list[i].comm_vol)
                    job_dag.remove_edges_from(
                        # Remove all edges with weight of 0 since we have no placeholder for "this edge doesn't exist"
                        [edge for edge in job_dag.edges() if job_dag.get_edge_data(*edge)['weight'] == '0.0']
                    )
                    nx.relabel_nodes(job_dag, lambda idx: idx + self.offset, copy=False)

                    computation_dict = common.computation_dict
                    power_dict = common.power_dict
                    outstanding_dag = common.current_dag

                    # Build the computation and power matrices that the scheduler will use to determine estimated execution time and power consumption
                    for node in job_dag:
                        computation_dict[node] = []
                        power_dict[node] = []
                        for cluster in common.ClusterManager.cluster_list:
                            cluster_power = cluster.current_power_cluster
                            for resource_idx in cluster.PE_list:
                                resource = self.resource_matrix.list[resource_idx]
                                associated_task = [task for task in self.generated_job_list[i].task_list if task.ID == node - self.offset]
                                if len(associated_task) > 0 and associated_task[0].name in resource.supported_functionalities:
                                    perf_index = resource.supported_functionalities.index(associated_task[0].name)
                                    computation_dict[node].append(resource.performance[perf_index])
                                    power_dict[node].append(cluster_power / len(cluster.PE_list))
                                else:
                                    computation_dict[node].append(np.inf)
                                    power_dict[node].append(np.inf)

                    # Build the current list of running or already-scheduled tasks so that the scheduler takes them into account
                    running_tasks = {}
                    for idx in range(len(self.PEs)):
                        running_tasks[idx] = []

                    for task in common.TaskQueues.running.list:
                        executing_resource = self.scheduler.resource_matrix.list[task.PE_ID]
                        task_id = task.ID
                        task_start = task.start_time
                        task_end = task_start + executing_resource.performance[executing_resource.supported_functionalities.index(task.name)]
                        proc = task.PE_ID
                        running_tasks[proc].append(heft.ScheduleEvent(task_id, task_start, task_end, proc))

                    for task in common.TaskQueues.executable.list:
                        executing_resource = self.scheduler.resource_matrix.list[task.PE_ID]
                        task_id = task.ID
                        # TODO: This is an implicit scheduling decision that the scheduler doesn't question
                        # The task hasn't started yet, so assume it will after the last task in this PE's queue
                        if len(running_tasks[task.PE_ID]) != 0:
                            task_start = running_tasks[task.PE_ID][-1].end
                        else:
                            task_start = self.env.now
                        task_end = task_start + executing_resource.performance[executing_resource.supported_functionalities.index(task.name)]
                        proc = task.PE_ID
                        running_tasks[proc].append(heft.ScheduleEvent(task_id, task_start, task_end, proc))

                    # Merge the DAG of the generated job with the current system DAG
                    merge_method = dag_merge.MergeMethod[common.config.get('HEFT SCHEDULER', 'heft_mergeMethod', fallback='COMMON_ENTRY_EXIT')]

                    if common.use_adaptive_scheduling:
                        if common.results.job_counter == common.max_jobs_in_parallel:
                            # System is oversubscribed, use EFT scheduling
                            rank_metric = heft.RankMetric.MEAN
                            op_mode = heft.OpMode.EFT
                        else:
                            # System isn't oversubscribed, use EDP scheduling
                            rank_metric = heft.RankMetric.EDP
                            op_mode = heft.OpMode.EDP_REL
                    else:
                        rank_metric = heft.RankMetric(common.config.get('HEFT SCHEDULER', 'heft_rankMetric', fallback='MEAN'))
                        op_mode = heft.OpMode(common.config.get('HEFT SCHEDULER', 'heft_opMode', fallback='EFT'))

                    # Remove placeholder source/sink nodes from the last time the graph was merged so they don't continually accumulate with each iteration
                    if len(outstanding_dag) != 0:
                        outstanding_dag.remove_node(max(outstanding_dag) - 1)
                        outstanding_dag.remove_node(max(outstanding_dag))

                    merged_dag = dag_merge.merge_dags(outstanding_dag, job_dag, merge_method=dag_merge.MergeMethod.COMMON_ENTRY_EXIT, skip_relabeling=True)

                    computation_dict[max(merged_dag) - 1] = np.zeros((1, len(self.scheduler.resource_matrix.list)))
                    computation_dict[max(merged_dag)] = np.zeros((1, len(self.scheduler.resource_matrix.list)))

                    power_dict[max(merged_dag) - 1] = np.zeros((1, len(self.scheduler.resource_matrix.list)))
                    power_dict[max(merged_dag)] = np.zeros((1, len(self.scheduler.resource_matrix.list)))

                    if common.DEBUG_SCH:
                        plt.clf()
                        # Requires "pydot" package, available through conda install.
                        nx.draw(merged_dag, pos=nx.nx_pydot.graphviz_layout(merged_dag, prog='dot'), with_labels=True)
                        plt.show()

                    # Load the appropriate parameters for passing this current DAG through HEFT
                    computation_matrix = np.empty((max(merged_dag) + 1, len(self.scheduler.resource_matrix.list)))  # Number of nodes * number of resources
                    for key, val in computation_dict.items():
                        computation_matrix[key, :] = val
                    power_matrix = np.empty((max(merged_dag) + 1, len(self.scheduler.resource_matrix.list)))  # Number of nodes * number of resources
                    for key, val in power_dict.items():
                        power_matrix[key, :] = val

                    if self.scheduler.name == 'HEFT':
                        proc_sched, task_sched, dict_output = heft.schedule_dag(
                            merged_dag,
                            computation_matrix=computation_matrix,
                            communication_matrix=common.ResourceManager.comm_band,
                            proc_schedules=running_tasks,
                            time_offset=self.env.now,
                            relabel_nodes=False,
                            rank_metric=rank_metric,
                            power_dict=power_matrix,
                            op_mode=op_mode
                        )
                    else:
                        proc_sched, task_sched, dict_output = peft.schedule_dag(
                            merged_dag,
                            computation_matrix=computation_matrix,
                            communication_matrix=common.ResourceManager.comm_band,
                            proc_schedules=running_tasks,
                            time_offset=self.env.now,
                            relabel_nodes=False
                        )

                    if common.DEBUG_SCH:
                        print('[D] Predicted HEFT finish time is %f' % task_sched[max(key for key, _ in task_sched.items())].end)
                        # gantt.showGanttChart(proc_sched)

                    common.table = dict_output
                    common.current_dag = merged_dag
                    common.computation_dict = computation_dict

                    # Update dynamic dependencies on tasks that are in the executable list to avoid deadlock situations
                    for executable_task in common.TaskQueues.executable.list:
                        executable_task.dynamic_dependencies = dict_output[executable_task.ID][2]
                # end if (scheduler.name == HEFT or scheduler.name == PEFT)

                for ii in range(len(self.generated_job_list[i].task_list)):                    # Go over each task in the job
                    next_task = self.generated_job_list[i].task_list[ii]
                    next_task.jobID = i                                         # assign job id to the next task
                    next_task.base_ID = ii                                      # also record the original ID of the next task
                    next_task.ID = ii + self.offset                             # and change the ID of the task accordingly
                    #### ACUMEN ###
                    if 'ACUMEN' in self.generated_job_list[i].name:
                        if i > 0 and (next_task.ID % 9 == 4):
                            #print(self.generated_job_list[i].name, next_task.ID, self.generated_job_list[i-1].task_list[-1].ID)
                            next_task.dynamic_dependencies.append(self.generated_job_list[i-1].task_list[-1].ID)  
                    #### ACUMEN ###        
                            
                    if next_task.head:
                        next_task.job_start = self.env.now                      # When a new job is generated, its execution is also started
                        common.results.total_execution_time = self.env.now
                        self.generated_job_list[i].head_ID = next_task.ID

                    next_task.head_ID = self.generated_job_list[i].head_ID

                    for k in range(len(next_task.predecessors)):
                        next_task.predecessors[k] += self.offset                # also change the predecessors of the newly added task, accordingly

                    if len(next_task.predecessors) > 0:
                        common.TaskQueues.outstanding.list.append(next_task)    # Add the task to the outstanding queue since it has predecessors
                        # Next, print debug messages
                        if (common.DEBUG_SIM):
                            print('[D] Time %d: Adding task %d to the outstanding queue,'
                                  % (self.env.now, next_task.ID), end='')
                            print(' task %d has predecessors:'
                                  % (next_task.ID), next_task.predecessors)
                    else:
                        common.TaskQueues.ready.list.append(next_task)          # Add the task to the ready queue since it has no predecessors
                        if (common.DEBUG_SIM):
                            print('[D] Time %s: Task %s is pushed to the ready queue list'
                                  % (self.env.now, next_task.ID), end='')
                            print(', the ready queue list has %s tasks'
                                  % (len(common.TaskQueues.ready.list)))
                self.offset += len(self.generated_job_list[i].task_list)
                # end of for ii in range(len(self.generated_job_list[i].list))

                if self.scheduler.name == 'CP':
                    while len(common.TaskQueues.executable.list) > 0:
                        task = common.TaskQueues.executable.list.pop(-1)
                        common.TaskQueues.ready.list.append(task)
                    
                    CP_models.CP(self.env.now, self.PEs, self.resource_matrix, self.jobs, self.generated_job_list)

                # Update the job ID
                i += 1
                if self.env.now >= common.warmup_period or common.simulation_mode == 'validation':
                    num_jobs += 1
                    if common.job_counter_list != []:
                        common.job_counter_list[selection] += 1
                        count_complete_jobs = 0
                        # Check if all jobs for the current snippet were injected
                        common.current_job_list = DASH_Sim_utils.get_current_job_list()
                        for index, job_counter in enumerate(common.job_counter_list):
                            if job_counter == common.current_job_list[index]:
                                count_complete_jobs += 1
                        if count_complete_jobs == len(common.job_counter_list) and num_jobs < common.max_num_jobs:
                            # Get the next snippet's job list
                            common.snippet_ID_inj += 1
                            np.random.seed(common.iteration)
                            common.job_counter_list = [0]*len(common.current_job_list)

                if (common.simulation_mode == 'validation' or common.inject_fixed_num_jobs):
                    if (num_jobs >= self.max_num_jobs):                                 # check if max number of jobs, given in config file, are created
                        self.generate_job = False                                       # if yes, no more jobs will be added to simulation
                
                
                # print ('lambda value is: %.2f' %(1/common.scale))
                if common.fixed_injection_rate:
                    self.wait_time = common.scale
                else:
                    self.wait_time = int(random.expovariate(1 / common.scale))      # assign an exponentially distributed random variable to $wait_time
                try:
                    yield self.env.timeout(self.wait_time)                          # new job addition will be after this wait time
                    #yield self.env.timeout(a_list[i%len(a_list)])
                except simpy.exceptions.Interrupt:
                    pass
            # end of while (self.generate_job):
