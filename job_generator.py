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
    def __init__(self, env, resource_matrix, jobs, scheduler, PE_list, resource_matrix_Acc):
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
        self.resource_matrix_Acc = resource_matrix_Acc
        
        
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
        # np.random.seed(common.iteration) commented out to see if it helps with the random seed issue

        
        if len(DASH_Sim_utils.get_current_job_list()) != len(self.jobs.list) and DASH_Sim_utils.get_current_job_list() != []:
            print('[E] Time %s: Job_list and job_file configs have different lengths, please check SoC.**.txt file'
                  % (self.env.now))
            sys.exit()

        if (self.scheduler.name == 'HEFT_RT_FUSION'):
            # Iterate over all jobs in the system and try to perform task fusion.
            for job in self.jobs.list:
                a = 0
                while a < len(job.task_list):
                    curr_task = job.task_list[a]
                    # Find successors of curr_task: tasks that have curr_task.ID as their only dependency.
                    successors = [t for t in job.task_list if (curr_task.ID in t.predecessors)]
                    # Check that there is exactly one successor and that it depends solely on curr_task.
                    if len(successors) == 1 and (len(successors[0].predecessors) == 1):
                        succ_task = successors[0]
                        # Count the number of successors for curr_task by scanning the task list.
                        curr_successors = [t for t in job.task_list if curr_task.ID in t.predecessors]
                        if len(curr_successors) == 1:
                            # Check SoC hardware compatibility: there must be at least one ACC resource that supports both tasks.
                            fusion_allowed = False
                            for resource in self.scheduler.resource_matrix.list:
                                if resource.type.upper() == "ACC":
                                    if (curr_task.name in resource.supported_functionalities and
                                        succ_task.name in resource.supported_functionalities):
                                        fusion_allowed = True
                                        # Add the new fused task to this resource’s supported functionalities if not already there.
                                        fused_config = curr_task.name + "_" + succ_task.name
                                        if fused_config not in resource.supported_functionalities:
                                            resource.supported_functionalities.append(fused_config)
                                            resource.num_of_functionalities += 1
                                            resource.DAP_config_list.append(fused_config)
                                            curr_task_exec_time = resource.performance[resource.supported_functionalities.index(curr_task.name)]
                                            succ_task_exec_time = resource.performance[resource.supported_functionalities.index(succ_task.name)]
                                            fused_task_exec_time = curr_task_exec_time + succ_task_exec_time
                                            resource.performance.append(fused_task_exec_time)
                                            # Add the new config and other informations about it into the ACC_model.csv file.
                                            import csv
                                            acc_csv_path = r"C:\\Users\\emre-\\DASH-Sim-Release-master\\ACC_model.csv"
                                            dynamic_power_curr = None
                                            curr_task_config = resource.DAP_config_list[resource.supported_functionalities.index(curr_task.name)]
                                            dynamic_power_succ = None
                                            succ_task_config = resource.DAP_config_list[resource.supported_functionalities.index(succ_task.name)]
                                            with open(acc_csv_path, "r", newline="") as f:
                                                reader = csv.DictReader(f)
                                                for row in reader:
                                                    if row["Config"] == curr_task_config:
                                                        dynamic_power_curr = float(row["Dynamic Power (W)"])
                                                    if row["Config"] == succ_task_config:
                                                        dynamic_power_succ = float(row["Dynamic Power (W)"])
                                            if dynamic_power_curr is None:
                                                dynamic_power_curr = 0
                                            if dynamic_power_succ is None:
                                                dynamic_power_succ = 0
                                            fused_dynamic_power = dynamic_power_curr + dynamic_power_succ
                                            formatted_fused_dynamic_power = round(fused_dynamic_power, 4) 

                                            if resource.name.startswith("DAP_0") or resource.name.startswith("DAP_1"):
                                                pe_util = 0.5
                                                dap_subpe = 16
                                                new_row = {
                                                    "PE Type": resource.name[:5],
                                                    "Config": fused_config,
                                                    "Throughput (sample/cycle)": "1",
                                                    "Data Transfer Latency": "1",
                                                    "Programming Latency": "0",
                                                    "PE Utilization": str(pe_util),
                                                    "Leakage Power (W)": "0",
                                                    "Dynamic Power (W)": str(formatted_fused_dynamic_power),
                                                    "DAP sub-PE": str(dap_subpe)
                                                }
                                                new_Acc_object = common.ResourceAcc()
                                                new_Acc_object.type = resource.name[:5]
                                                new_Acc_object.programming_latency = 0
                                                new_Acc_object.leakage_power = float(0)
                                                new_Acc_object.dynamic_power = formatted_fused_dynamic_power
                                                new_Acc_object.config = fused_config
                                                new_Acc_object.DAP_subPEs = dap_subpe
                                                self.resource_matrix_Acc.dict[resource.name[:5] + "," + fused_config] = new_Acc_object
                                                common.resource_matrix_Acc = self.resource_matrix_Acc
                                            else:
                                                pe_util = 1
                                                dap_subpe = 0
                                                new_row = {
                                                    "PE Type": resource.name[:3],
                                                    "Config": fused_config,
                                                    "Throughput (sample/cycle)": "1",
                                                    "Data Transfer Latency": "1",
                                                    "Programming Latency": "0",
                                                    "PE Utilization": str(pe_util),
                                                    "Leakage Power (W)": "0",
                                                    "Dynamic Power (W)": str(formatted_fused_dynamic_power),
                                                    "DAP sub-PE": str(dap_subpe)
                                                }
                                                new_Acc_object = common.ResourceAcc()
                                                new_Acc_object.type = resource.name[:3]
                                                new_Acc_object.programming_latency = 0
                                                new_Acc_object.leakage_power = float(0)
                                                new_Acc_object.dynamic_power = formatted_fused_dynamic_power
                                                new_Acc_object.config = fused_config
                                                new_Acc_object.DAP_subPEs = dap_subpe
                                                self.resource_matrix_Acc.dict[resource.name[:3] + "," + fused_config] = new_Acc_object
                                                common.resource_matrix_Acc = self.resource_matrix_Acc

                                            with open(acc_csv_path, "a", newline="") as f:
                                                writer = csv.DictWriter(f, fieldnames=["PE Type",
                                                                                        "Config",
                                                                                        "Throughput (sample/cycle)",
                                                                                        "Data Transfer Latency",
                                                                                        "Programming Latency",
                                                                                        "PE Utilization",
                                                                                        "Leakage Power (W)",
                                                                                        "Dynamic Power (W)",
                                                                                        "DAP sub-PE"])
                                                writer.writerow(new_row)
                 
                            if fusion_allowed:
                                # Create new fused task.
                                fused_task = type(curr_task)()  
                                fused_task.name = curr_task.name + "_" + succ_task.name
                                # Use the lower id (curr_task) as the fused task id.
                                fused_task.ID = curr_task.ID
                                fused_task.deadline = max(curr_task.deadline, succ_task.deadline)
                                fused_task.dag_depth = curr_task.dag_depth
                                fused_task.est = curr_task.est
                                fused_task.input_packet_size = curr_task.input_packet_size
                                fused_task.output_packet_size = curr_task.output_packet_size
                                fused_task.jobname = curr_task.jobname
                                fused_task.base_ID = curr_task.base_ID
                                # Predecessors of the fused task come from the predecessor(s) of curr_task.
                                fused_task.predecessors = curr_task.predecessors.copy()
                                fused_task.preds = curr_task.preds.copy()

                                # Save the communication volumes from the predecessor(s) of the curr_task.
                                row_from_pred_of_curr_task = {}
                                for t in curr_task.predecessors:
                                    row_from_pred_of_curr_task[t] = job.comm_vol[t, :].copy()

                                succ_id = succ_task.ID
                                fused_id = fused_task.ID 
                                
                                # Remove succ_task from the job's task list and update the current task slot.
                                job.task_list[a] = fused_task
                                job.task_list.remove(succ_task)

                                # Update the IDs and base IDs of the subsequent tasks in the task list.
                                for t in job.task_list:
                                   if t.ID > succ_id:
                                       t.ID -= 1
                                       t.base_ID -= 1 

                                # Update dependencies in other tasks: if any task had succ_task as a predecessor, replace it with fused_task.ID.
                                for t in job.task_list:
                                    for j in range(len(t.predecessors)):
                                        if t.predecessors[j] == succ_id:
                                            t.predecessors[j] = fused_id
                                            t.preds[j] = fused_id
                                        if t.predecessors[j] > succ_id:
                                            t.predecessors[j] -= 1
                                            t.preds[j] -= 1   

                                if (common.DEBUG_SCH):
                                    print("[D] Fused tasks %d and %d into %s" % (curr_task.ID, succ_task.ID, fused_task.name))

                                row_from_succ = job.comm_vol[succ_id, :].copy() # Update the row of the fused task in the comm_vol matrix.
                                job.comm_vol[fused_id, :] = row_from_succ
                                col_from_succ = job.comm_vol[:, succ_id].copy() # Update the column of the fused task in the comm_vol matrix.
                                job.comm_vol[:, fused_id] = col_from_succ

                                # Update the communication volumes from the predecessor(s) of fused_task.
                                for k, row in row_from_pred_of_curr_task.items():
                                    job.comm_vol[k, :] = row

                                job.comm_vol = np.delete(job.comm_vol, succ_id, axis=0) # Delete the row of the succ_task.
                                job.comm_vol = np.delete(job.comm_vol, succ_id, axis=1) # Delete the column of the succ_task.    

                                if succ_id in job.dag_depth:
                                    del job.dag_depth[succ_id] # Delete the succ_task from the dag_depth dictionary.

                                updated_dag_depth = {} # Update the dag_depth dictionary according to the recent changes.
                                for key, depth in job.dag_depth.items():
                                    if key == 'DAG':
                                        updated_dag_depth[key] = depth
                                    elif isinstance(key, int):
                                        if key > succ_id:
                                            updated_dag_depth[key - 1] = depth
                                        else:
                                            updated_dag_depth[key] = depth
                                    else:
                                        updated_dag_depth[key] = depth
                                
                                job.dag_depth = updated_dag_depth

                                individual_depths = [d for k, d in job.dag_depth.items() if k != 'DAG']
                                # Build a sorted list of unique depths.
                                unique_depths = sorted(set(individual_depths))

                                # Create a mapping from the unique depths to new depths to get rid of the jumps in the dag_depth dictionary, if there is any.
                                new_depth_map = {orig_depth: new_depth for new_depth, orig_depth in enumerate(unique_depths)}

                                # Update every task's dag_depth stored in job.dag_depth (excluding the 'DAG' key) using the mapping.
                                for key in list(job.dag_depth.keys()):
                                    if key != 'DAG':
                                        job.dag_depth[key] = new_depth_map[job.dag_depth[key]]

                                # Finally, update the overall 'DAG' depth as the maximum of the new values.
                                job.dag_depth['DAG'] = max(new_depth_map.values()) if new_depth_map else 0
                                # Do not increment a so we can check if further fusion is possible on the new fused task.
                                continue
                    a += 1
        # End of HEFT_RT_FUSION fusion block

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
                        #   np.random.seed(common.iteration) commented out to see if it helps with the random seed issue
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
