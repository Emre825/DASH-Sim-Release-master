'''!
@brief This file contains the simulation core that handles the simulation events.
'''
import sys
import numpy as np

import common                                                                   # The common parameters used in DASH-Sim are defined in common_parameters.py
import random
import DTPM
import DTPM_policies
import DASH_acc_utils
import DASH_Sim_utils
import DTPM_power_models

# If the user specifies the latency model between each PE, it can be imported.
if common.latency_matrix:
    import NoC_model

# Define the core of the simulation engine
# This function calls the scheduler, starts/interrupts the tasks,
# and manages collection of all the statistics

class SimulationManager:
    '''!
    Define the SimulationManager class to handle the simulation events.
    '''
    def __init__(self, env, sim_done, job_gen, scheduler, PE_list, jobs, resource_matrix):
        '''!
        @param env: Pointer to the current simulation environment
        @param sim_done: Simpy event object to indicate whether the simulation must be finished
        @param job_gen: JobGenerator object
        @param scheduler: Pointer to the DASH_scheduler
        @param PE_list: The PEs available in the current SoC
        @param jobs: The list of all jobs given to DASH-Sim
        @param resource_matrix: The data structure that defines power/performance characteristics of the PEs for each supported task
        '''
        self.env = env
        self.sim_done = sim_done
        self.job_gen = job_gen
        self.scheduler = scheduler
        self.PEs = PE_list
        self.jobs = jobs
        self.resource_matrix = resource_matrix

        self.action = env.process(self.run())  # starts the run() method as a SimPy process



    def update_ready_queue(self,completed_task):
        '''!
        This function updates the common.TaskQueues.ready after one task is completed.

        As the simulation proceeds, tasks are being processed.
        We need to update the ready tasks queue after completion of each task.

        @param completed_task: Object for the task that just completed execution
        '''

        # completed_task is the task whose processing is just completed
        # Add completed task to the completed tasks queue
        common.TaskQueues.completed.list.append(completed_task)

        # Remove the completed task from the queue of the PE
        for task in self.PEs[completed_task.PE_ID].queue:
            if task.ID == completed_task.ID:
                self.PEs[task.PE_ID].queue.remove(task)

        # Remove the completed task from the currently running queue
        common.TaskQueues.running.list.remove(completed_task)

        # Remove the completed task from the current DAG representation
        if completed_task.ID in common.current_dag:
            common.current_dag.remove_node(completed_task.ID)
        
        # Initialize $remove_from_outstanding_queue which will populate tasks
        # to be removed from the outstanding queue
        remove_from_outstanding_queue = []

        # Initialize $to_memory_comm_time which will be communication time to
        # memory for data from a predecessor task to a outstanding task
        to_memory_comm_time = -1

        if (common.latency_matrix):
            # Send the output data of the completed task from PE to cache
            cache_position = int(np.random.choice(NoC_model.Cache_positions, 1, p=[0.25, 0.25, 0.25, 0.25]))
            PE_position = int(self.resource_matrix.list[completed_task.PE_ID].position)
            buffer_ind = int(self.env.now % 100)
            # print(PE_position,cache_position,buffer_ind)

            common.PE_to_Cache[(PE_position, cache_position)] += int(completed_task.output_packet_size)
            NoC_model.write_to_cache(self.env.now, PE_position, cache_position, buffer_ind, completed_task)
        
        job_ID = -1
        for ind, job in enumerate(self.jobs.list):
            if job.name == completed_task.jobname:
                job_ID = ind


        # Check if the dependency of any outstanding task is cleared
        # We need to move them to the ready queue
        for i, outstanding_task in enumerate(common.TaskQueues.outstanding.list):                       # Go over each outstanding task
            if (completed_task.ID in outstanding_task.predecessors):                                    # if the completed task is one of the predecessors
                outstanding_task.predecessors.remove(completed_task.ID)                                 # Clear this predecessor

                if (common.shared_memory):
                    # Get the communication time to memory for data from a
                    # predecessor task to a outstanding task
                    comm_vol = self.jobs.list[job_ID].comm_vol[completed_task.base_ID , outstanding_task.base_ID]
                    comm_band = common.ResourceManager.comm_band[completed_task.PE_ID, self.resource_matrix.list[-1].ID]
                    to_memory_comm_time = int(comm_vol/comm_band)                                           # Communication time from a PE to memory

                    if (common.DEBUG_SIM):
                        print('[D] Time %d: Data from task %d for task %d will be sent to memory in %d us'
                              %(self.env.now, completed_task.ID, outstanding_task.ID, to_memory_comm_time))

                    # Based on this communication time, this outstanding task
                    # will be added to the ready queue. That is why, keep track of
                    # all communication times required for a task in the list
                    # $ready_wait_times
                    outstanding_task.ready_wait_times.append(to_memory_comm_time + self.env.now)
                # end of if (common.shared_memory):

                if (common.latency_matrix):
                    # Get the latency for writing the data to the cache
                    # print(cache_position)
                    write_latency = NoC_model.late_matrix[PE_position][cache_position]
                    # print(write_latency)

                    if (common.DEBUG_SIM):
                        print('[D] Time %d: Data from task %d for task %d will be sent to memory in %d us'
                              % (self.env.now, completed_task.ID, outstanding_task.ID, write_latency))

                    # one task may have more than one predecessor
                    # for this reason, all write latencies from these predecessors will
                    # be populated in $ready_wait_times
                    outstanding_task.ready_wait_times.append(int(write_latency) + self.env.now)
                # end of if (common.latency_matrix):
            # end of if (completed_task.ID in outstanding_task.predecessors):

            no_predecessors = (len(outstanding_task.predecessors) == 0)                            # Check if this was the last dependency
            currently_running = (outstanding_task in                                               # if the task is in the running queue,
                                 common.TaskQueues.running.list)                                   # We should not put it back to the ready queue
            not_in_ready_queue = not(outstanding_task in                                           # If this task is already in the ready queue,
                                  common.TaskQueues.ready.list)                                    # We should not append another copy

            if (no_predecessors and not(currently_running) and not_in_ready_queue):
                if (common.PE_to_PE):                                                              # if PE to PE communication is utilized
                    common.TaskQueues.ready.list.append(common.TaskQueues.outstanding.list[i])     # Add the task to the ready queue immediately

                elif (common.shared_memory):
                    # if shared memory is utilized for communication, then
                    # the outstanding task will wait for a certain amount time
                    # (till the $time_stamp)for being added into the ready queue
                    common.TaskQueues.wait_ready.list.append(outstanding_task)
                    if (common.INFO_SIM) and (common.shared_memory):
                            print('[I] Time %d: Task %d ready times due to memory communication of its predecessors are'
                                  %(self.env.now, outstanding_task.ID))
                            print('%12s'%(''), outstanding_task.ready_wait_times)
                    common.TaskQueues.wait_ready.list[-1].time_stamp = max(outstanding_task.ready_wait_times)

                elif (common.latency_matrix):
                    # the outstanding task will wait for a certain amount time
                    # (till the $time_stamp) for being added into the ready queue
                    common.TaskQueues.wait_ready.list.append(outstanding_task)
                    if (common.INFO_SIM):
                        print('[I] Time %d: Task %d ready times due to writing the data to cache are'
                              %(self.env.now, outstanding_task.ID))
                        print('%12s'%(''), outstanding_task.ready_wait_times)
                    common.TaskQueues.wait_ready.list[-1].time_stamp = max(outstanding_task.ready_wait_times)

                remove_from_outstanding_queue.append(outstanding_task)
        # end of for i, outstanding_task in...

        # Remove the tasks from outstanding queue that have been moved to ready queue
        for task in remove_from_outstanding_queue:
            common.TaskQueues.outstanding.list.remove(task)

        # At the end of this function:
            # Newly processed $completed_task is added to the completed tasks
            # outstanding tasks with no dependencies are added to the ready queue
            # based on the communication mode and then, they are removed from
            # the outstanding queue
    #end def update_ready_queue(completed_task)

    def update_execution_queue(self, ready_list):
        '''!
        This function updates the common.TaskQueues.executable if one task is ready
        for execution but waiting for the communication time, either between
        memory and a PE, or between two PEs (based on the communication mode)

        @param ready_list: List of tasks that are ready to be executed
        '''
        # Initialize $remove_from_ready_queue which will populate tasks
        # to be removed from the outstanding queue
        remove_from_ready_queue = []
        
        # Initialize $from_memory_comm_time which will be communication time 
        # for data from memory to a PE
        from_memory_comm_time = -1

        # Initialize $PE_to_PE_comm_time which will be communication time
        # for data from a PE to another PE
        PE_to_PE_comm_time = -1

        job_ID = -1
        for ready_task in ready_list:
            if (common.latency_matrix):  # If NoC Analytical mode used (latency_matrix communication mode)

                # Compute the read latency from cache or main memory based on cache hit/miss
                cache_hit = np.random.choice([True, False], 1, p=[0.8, 0.2])
                cacti_cache_access = 1
                cacti_dram_access = 1
                cache_position = int(np.random.choice(NoC_model.Cache_positions, 1, p=[0.25, 0.25, 0.25, 0.25]))
                PE_position = int(self.resource_matrix.list[ready_task.PE_ID].position)
                buffer_ind = int(self.env.now % 100)

                NoC_model.read_from_cache(cache_hit, self.env.now, cache_position, PE_position, buffer_ind, ready_task)

                ind = self.resource_matrix.list[ready_task.PE_ID].supported_functionalities.index(ready_task.name)
                if self.resource_matrix.list[ready_task.PE_ID].performance[ind] == 0:
                    ready_task.execution_wait_times.append(0 + self.env.now)
                else:
                    if (cache_hit):
                        # print('yes', cache_hit)
                        read_latency = NoC_model.late_matrix[cache_position][ready_task.PE_ID] + cacti_cache_access

                        if (common.DEBUG_SIM):
                            print('[D] Time %d: Data from cache for task %d will be sent to PE-%s in %d us'
                                  % (self.env.now, ready_task.ID, ready_task.PE_ID, read_latency))
                        ready_task.execution_wait_times.append(read_latency + self.env.now)
                    else:  # No hit
                        # print('no', cache_hit)
                        MC_latency_1 = NoC_model.late_matrix[cache_position][NoC_model.MC_position]
                        MC_latency_2 = NoC_model.late_matrix[NoC_model.MC_position][cache_position]
                        cache_to_PE = NoC_model.late_matrix[cache_position][ready_task.PE_ID]

                        read_latency = MC_latency_1 + cacti_dram_access + MC_latency_2 + cacti_cache_access + cache_to_PE

                        if (common.DEBUG_SIM):
                            print('[D] Time %d: Data from main memory for task %d will be sent to PE-%s in %d us'
                                  % (self.env.now, ready_task.ID, ready_task.PE_ID, read_latency))
                        ready_task.execution_wait_times.append(read_latency + self.env.now)
                # end of if self.resource_matrix.list[ready_task.PE_ID].performance[ind] == 0:

                if (common.INFO_SIM) and (common.latency_matrix):
                    print('[I] Time %d: Task %d execution ready time due to communication between cache/dram and PE-%s is %s'
                          % (self.env.now, ready_task.ID, ready_task.PE_ID, max(ready_task.execution_wait_times)))

                common.TaskQueues.executable.list.append(ready_task)
                remove_from_ready_queue.append(ready_task)
                common.TaskQueues.executable.list[-1].time_stamp = max(ready_task.execution_wait_times)
            # end of if (common.latency_matrix):

            else:  # If other communication modes are used (PE_to_PE or shared_memory)
                for ind, job in enumerate(self.jobs.list):
                    if job.name == ready_task.jobname:
                        job_ID = ind

                for i, task in enumerate(self.jobs.list[job_ID].task_list):
                    if ready_task.base_ID == task.ID:
                        if ready_task.head == True:
                            # if a task is the leading task of a job
                            # then it can start immediately since it has no predecessor
                            ready_task.PE_to_PE_wait_time.append(self.env.now)
                            ready_task.execution_wait_times.append(self.env.now)
                        # end of if ready_task.head == True:

                        for predecessor in task.predecessors:
                            if(task.ID==ready_task.ID):
                               ready_task.predecessors = task.predecessors

                            # data required from the predecessor for $ready_task
                            comm_vol = self.jobs.list[job_ID].comm_vol[predecessor, ready_task.base_ID]

                            # retrieve the real ID  of the predecessor based on the job ID
                            real_predecessor_ID = predecessor + ready_task.ID - ready_task.base_ID

                            # Initialize following two variables which will be used if
                            # PE to PE communication is utilized
                            predecessor_PE_ID = -1
                            predecessor_finish_time = -1

                            if (common.PE_to_PE):
                                # Compute the PE to PE communication time
                                for completed in common.TaskQueues.completed.list:
                                    if completed.ID == real_predecessor_ID:
                                        predecessor_PE_ID = completed.PE_ID
                                        predecessor_finish_time = completed.finish_time
                                comm_band = common.ResourceManager.comm_band[predecessor_PE_ID, ready_task.PE_ID]

                                if common.comm_model == 'PE' :
                                    PE_to_PE_comm_time = int(comm_vol/comm_band)
                                elif common.comm_model == 'NoC' :
                                    ## Compute communication time
                                    ## If source and destination are same PE, then communication time is assumed to be zero
                                    ## If not, compute the time as the number of Manhattan hops on the 2D-mesh
                                    pred_cluster = self.resource_matrix.list[predecessor_PE_ID].cluster_ID
                                    task_cluster = self.resource_matrix.list[ready_task.PE_ID].cluster_ID

                                    if pred_cluster == task_cluster :
                                        PE_to_PE_comm_time = 0
                                    else :
                                        ## Get source and destination mesh positions and x,y indices
                                        src_mesh_pos = self.resource_matrix.list[predecessor_PE_ID].position
                                        dst_mesh_pos = self.resource_matrix.list[ready_task.PE_ID].position

                                        src_mesh_x   = int(src_mesh_pos % common.NoC_x_dim)
                                        src_mesh_y   = int(src_mesh_pos / common.NoC_x_dim)

                                        dst_mesh_x   = int(dst_mesh_pos % common.NoC_x_dim)
                                        dst_mesh_y   = int(dst_mesh_pos / common.NoC_x_dim)

                                        random_cache_slice = random.randint(0, len(common.cache_positions) - 1)
                                        cache_PE_ID    = common.cache_positions[random_cache_slice][0]
                                        cache_mesh_pos = common.cache_positions[random_cache_slice][1]

                                        cache_mesh_x   = int(cache_mesh_pos % common.NoC_x_dim)
                                        cache_mesh_y   = int(cache_mesh_pos / common.NoC_x_dim)

                                        num_hops = 2 +  \
                                                   abs(src_mesh_x - cache_mesh_x) + \
                                                   abs(src_mesh_y - cache_mesh_y) + \
                                                   abs(dst_mesh_x - cache_mesh_x) + \
                                                   abs(dst_mesh_y - cache_mesh_y)

                                        PE_to_PE_comm_time = num_hops + (ready_task.input_packet_size - 1)

                                        # print('INPUT:', ready_task.input_packet_size)
                                        # print('PrePE:', predecessor_PE_ID)
                                        # print('PreNm:', self.resource_matrix.list[predecessor_PE_ID].name)
                                        # print('CurPE:', ready_task.PE_ID)
                                        # print('CurNm:', self.resource_matrix.list[ready_task.PE_ID].name)
                                        # print('SRC-X:', src_mesh_x)
                                        # print('SRC-Y:', src_mesh_y)
                                        # print('DST-X:', dst_mesh_x)
                                        # print('DST-Y:', dst_mesh_y)
                                        # print('CAC-X:', cache_mesh_x)
                                        # print('CAC-Y:', cache_mesh_y)
                                        # print('NumHP:', PE_to_PE_comm_time)
                                        # print(abs(src_mesh_x - cache_mesh_x))
                                        # print(abs(src_mesh_y - cache_mesh_y))
                                        # print(abs(dst_mesh_x - cache_mesh_x))
                                        # print(abs(dst_mesh_y - cache_mesh_y))
                                        # print('-------------------')
                                else :
                                    print('[E] Incorrect communication model specified in configuration file')
                                    print('[E] Please use either "PE" or "NoC" for the communication model')
                                    exit()
                                ## if common.comm_model == 'PE' :
                                ready_task.PE_to_PE_wait_time.append(PE_to_PE_comm_time + predecessor_finish_time)

                                if (common.DEBUG_SIM):
                                    print('[D] Time %d: Data transfer from PE-%s to PE-%s for task %d from task %d is completed at %d us'
                                          %(self.env.now, predecessor_PE_ID, ready_task.PE_ID,
                                            ready_task.ID, real_predecessor_ID, ready_task.PE_to_PE_wait_time[-1]))
                            # end of if (common.PE_to_PE):

                            if (common.shared_memory):
                                # Compute the memory to PE communication time
                                comm_band = common.ResourceManager.comm_band[self.resource_matrix.list[-1].ID, ready_task.PE_ID]
                                from_memory_comm_time = int(comm_vol/comm_band)
                                if (common.DEBUG_SIM):
                                    print('[D] Time %d: Data from memory for task %d from task %d will be sent to PE-%s in %d us'
                                          %(self.env.now, ready_task.ID, real_predecessor_ID, ready_task.PE_ID, from_memory_comm_time))
                                ready_task.execution_wait_times.append(from_memory_comm_time + self.env.now)
                            # end of if (common.shared_memory)
                        # end of for predecessor in task.predecessors:

                        if (common.INFO_SIM) and (common.PE_to_PE):
                            print('[I] Time %d: Task %d execution ready times due to communication between PEs are'
                                  %(self.env.now, ready_task.ID))
                            print('%12s'%(''), ready_task.PE_to_PE_wait_time)

                        if (common.INFO_SIM) and (common.shared_memory):
                            print('[I] Time %d: Task %d execution ready time(s) due to communication between memory and PE-%s are'
                                  %(self.env.now, ready_task.ID, ready_task.PE_ID))
                            print('%12s'%(''), ready_task.execution_wait_times)

                        # Populate all ready tasks in executable with a time stamp
                        # which will show when a task is ready for execution
                        common.TaskQueues.executable.list.append(ready_task)
                        remove_from_ready_queue.append(ready_task)
                        if (common.PE_to_PE):
                            common.TaskQueues.executable.list[-1].time_stamp = max(ready_task.PE_to_PE_wait_time) + common.TaskQueues.executable.list[-1].scheduling_overhead
                        else:
                            common.TaskQueues.executable.list[-1].time_stamp = max(ready_task.execution_wait_times)
                # end of ready_task.base_ID == task.ID:
            # end of i, task in enumerate(self.jobs.list[job_ID].task_list):    
        # end of for ready_task in ready_list:
        
        # Remove the tasks from ready queue that have been moved to executable queue
        for task in remove_from_ready_queue:
            common.TaskQueues.ready.list.remove(task)

        if self.scheduler.name != 'HEFT' and self.scheduler.name != 'HEFT_RT' and self.scheduler.name != 'PEFT' and self.scheduler.name != 'PEFT_RT':
            # Reorder tasks based on their job IDs
            common.TaskQueues.executable.list.sort(key=lambda task: task.jobID, reverse=False)
            
    def update_completed_queue(self):
        '''!
        This function updates the common.TaskQueues.completed 
        '''  
        ## Be careful about this function when there are diff jobs in the system
        # reorder tasks based on their job IDs
        common.TaskQueues.completed.list.sort(key=lambda x: x.jobID, reverse=False)
        
        first_task_jobID =  common.TaskQueues.completed.list[0].jobID
        last_task_jobID = common.TaskQueues.completed.list[-1].jobID
        
        if ((last_task_jobID - first_task_jobID) > 15):
            for i,task in enumerate(common.TaskQueues.completed.list):
                if (task.jobID == first_task_jobID):
                    del common.TaskQueues.completed.list[i]
            
        
    #
    def run(self):
        '''!
        Implement the basic run method that will be called periodically in each simulation "tick".

        This function takes the next ready tasks and run on the specific PE and update the common.TaskQueues.ready list accordingly.
        '''
        DTPM_module = DTPM.DTPMmodule(self.env, self.resource_matrix, self.PEs)

        for cluster in common.ClusterManager.cluster_list:
            DTPM_policies.initialize_frequency(cluster)

        while (True):                                                           # Continue till the end of the simulation

            if self.env.now % common.sampling_rate == 0:
                #common.results.job_counter_list.append(common.results.job_counter)
                #common.results.sampling_rate_list.append(self.env.now)
                # Evaluate idle PEs, busy PEs will be updated and evaluated from the PE class
                DTPM_module.evaluate_idle_PEs()
            # end of if self.env.now % common.sampling_rate == 0:

            if common.latency_matrix:
                if (self.env.now == common.warmup_period):
                    for row_idx in range(NoC_model.num_rows * NoC_model.num_cols):
                        for col_idx in range(NoC_model.num_rows * NoC_model.num_cols):
                            NoC_model.lambda_array[row_idx][col_idx] = sum(NoC_model.S2D_buffers[(row_idx, col_idx)]) / 100000

                    print(NoC_model.lambda_array)
                    NoC_model.lib_analytical_models.wrapperLatencyModels(NoC_model.num_rows, NoC_model.num_cols, NoC_model.lambda_array, NoC_model.late_matrix)
                    NoC_model.late_matrix = NoC_model.late_matrix / 1000
                    print(NoC_model.late_matrix)

            if (common.shared_memory) or (common.latency_matrix):
                # this section is activated only if shared memory or common.latency_matrix are used

                # Initialize $remove_from_wait_ready which will populate tasks
                # to be removed from the wait ready queue
                remove_from_wait_ready = []

                for i, waiting_task in enumerate(common.TaskQueues.wait_ready.list):
                    if waiting_task.time_stamp <= self.env.now:
                        common.TaskQueues.ready.list.append(waiting_task)
                        remove_from_wait_ready.append(waiting_task)
                # at the end of this loop, all the waiting tasks with a time stamp
                # equal or smaller than the simulation time will be added to
                # the ready queue list
                #end of for i, waiting_task in...

                # Remove the tasks from wait ready queue that have been moved to ready queue
                for task in remove_from_wait_ready:
                    common.TaskQueues.wait_ready.list.remove(task)
            # end of if (common.shared_memory):

            if (common.INFO_SIM) and len(common.TaskQueues.ready.list) > 0:
                print('[I] Time %s: DASH-Sim ticks with %d task ready for being assigned to a PE'
                      % (self.env.now, len(common.TaskQueues.ready.list)))

            if (not len(common.TaskQueues.ready.list) == 0):
                # give all tasks in ready_list to the chosen scheduler
                # and scheduler will assign the tasks to a PE
                if self.scheduler.name == 'CPU_only':
                    self.scheduler.CPU_only(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'MET':
                    self.scheduler.MET(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'EFT':
                    self.scheduler.EFT(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'STF':
                    self.scheduler.STF(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'ETF':
                    self.scheduler.ETF(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'ILS_ETF':
                    self.scheduler.ILS_ETF(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'ETF_LB':
                    self.scheduler.ETF_LB(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'HEFT':
                    self.scheduler.HEFT(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'HEFT_RT':
                    self.scheduler.HEFT_RT(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'PEFT':
                    self.scheduler.PEFT(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'PEFT_RT':
                    self.scheduler.PEFT_RT(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'CP':
                    self.scheduler.CP(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'DAS':
                    self.scheduler.DAS(common.TaskQueues.ready.list)                    
                else:
                    print('[E] Could not find the requested scheduler')
                    print('[E] Please check "config_file.ini" and enter a proper name')
                    print('[E] or check "scheduler.py" if the scheduler exist')
                    sys.exit()
                # end of if self.scheduler.name
                self.update_execution_queue(common.TaskQueues.ready.list)       # Update the execution queue based on task's info
            # end of if not len(common.TaskQueues.ready.list) == 0:

            # Initialize $remove_from_executable which will populate tasks
            # to be removed from the executable queue
            remove_from_executable = []

            # Go over each task in the executable queue
            if len(common.TaskQueues.executable.list) != 0:
                # for PE blocking data collection
                if self.env.now >= common.warmup_period:
                    for PE in self.PEs:
                        a_list = []
                        if not PE.idle:
                            for k, executable_task in enumerate(common.TaskQueues.executable.list):
                                if executable_task.PE_ID == PE.ID:
                                    if executable_task.time_stamp <= self.env.now:
                                        a_list.append(executable_task)
                            
                        if len(a_list) > 0:            
                            PE.blocking += 1
                
                for i, executable_task in enumerate(common.TaskQueues.executable.list):
                    is_time_to_execute = (executable_task.time_stamp <= self.env.now)
                    PE_has_capacity = DASH_Sim_utils.check_PE_capacity(self.PEs[executable_task.PE_ID], self.resource_matrix.list[executable_task.PE_ID], executable_task)
                    task_has_assignment = (executable_task.PE_ID != -1)

                    dynamic_dependencies_met = True

                    dependencies_completed = []
                    for dynamic_dependency in executable_task.dynamic_dependencies:
                        dependencies_completed = dependencies_completed + list(filter(lambda completed_task: completed_task.ID == dynamic_dependency, common.TaskQueues.completed.list))
                    if len(dependencies_completed) != len(executable_task.dynamic_dependencies):
                        dynamic_dependencies_met = False

                    if is_time_to_execute and PE_has_capacity and dynamic_dependencies_met and task_has_assignment:
                        self.PEs[executable_task.PE_ID].queue.append(executable_task)

                        # If executing on an accelerator, update the configuration (add)
                        DASH_acc_utils.add_config_acc(self.PEs[executable_task.PE_ID], self.resource_matrix.list[executable_task.PE_ID], executable_task)
                        if self.PEs[executable_task.PE_ID].type == 'ACC':
                            num_tasks = DASH_Sim_utils.get_num_tasks_being_executed(common.ClusterManager.cluster_list[self.PEs[executable_task.PE_ID].cluster_ID], self.PEs)
                            DTPM_power_models.set_active_cores(common.ClusterManager.cluster_list[self.PEs[executable_task.PE_ID].cluster_ID], self.PEs, num_tasks)

                        if (common.INFO_SIM):
                            print('[I] Time %s: Task %s is ready for execution by PE-%s'
                                  % (self.env.now, executable_task.ID, executable_task.PE_ID))

                        current_resource = self.resource_matrix.list[executable_task.PE_ID]
                        self.env.process(self.PEs[executable_task.PE_ID].run(  # Send the current task and a handle for this simulation manager (self)
                            self, executable_task, current_resource, DTPM_module))  # This handle is used by the PE to call the update_ready_queue function

                        remove_from_executable.append(executable_task)
                    # end of if is_time_to_execute and PE_has_capacity and dynamic_dependencies_met
                # end of for i, executable_task in...
            # end of if not len(common.TaskQueues.executable.list) == 0:

            # Remove the tasks from executable queue that have been executed by a resource
            for task in remove_from_executable:
                common.TaskQueues.executable.list.remove(task)

            # The simulation tick is completed. Wait till the next interval
            yield self.env.timeout(common.simulation_clk)

            # Reset the reconfiguration overhead (already incorporated by the PEs)
            for i in range(len(common.ClusterManager.cluster_list) - 1):
                common.ClusterManager.cluster_list[i].reconfiguration_overhead = 0

            if self.env.now > common.simulation_length and common.inject_fixed_num_jobs is False:
                self.sim_done.succeed()
        #end while (True)
