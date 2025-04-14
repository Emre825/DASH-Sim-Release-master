'''!
@brief This file contains all schedulers in DASH-Sim

Scheduler class is defined in this file which contains different types of scheduler as a member function.
Developers can add thier own algorithms here and implement in DASH-Sim by add a function caller in DASH_Sim_core.py
'''
import networkx as nx
import numpy as np
import copy

import common                                                                   # The common parameters used in DASH-Sim are defined in common_parameters.py
import random
from heft import heft, dag_merge
from peft import peft
import DTPM_power_models
import ils_scripts
import DASH_acc_utils
import DASH_Sim_utils

import pickle

class Scheduler:
    '''!
    The Scheduler class constains all schedulers implemented in DASH-Sim
    '''
    def __init__(self, env, resource_matrix, name, PE_list, jobs):
        '''!
        @param env: Pointer to the current simulation environment
        @param resource_matrix: The data structure that defines power/performance characteristics of the PEs for each supported task
        @param name : The name of the requested scheduler
        @param PE_list: The PEs available in the current SoCs
        @param jobs: The list of all jobs given to DASH-Sim
        '''
        self.env = env
        self.resource_matrix = resource_matrix
        self.name = name
        self.PEs = PE_list
        self.jobs = jobs
        self.assigned = [0] * (len(self.PEs))

        # At the end of this function, the scheduler class has a copy of the
        # the power/performance characteristics of the resource matrix and
        # name of the requested scheduler name


    # end  def __init__(self, env, resource_matrix, scheduler_name)

    # Specific scheduler instances can be defined below
    def CPU_only(self, list_of_ready):
        '''!
        This scheduler always select the resource with ID 0 (CPU) to execute all outstanding tasks without any comparison between
        available resources
        @param list_of_ready: The list of ready tasks
        '''
        for task in list_of_ready:
            task.PE_ID = 0

    # end def CPU_only(list_of_ready):


    def MET(self, list_of_ready):
        '''!
        This scheduler compares the execution times of the current task for available resources and returns the ID of the resource
        with minimum execution time for the current task.
        @param list_of_ready: The list of ready tasks
        '''
        # Initialize a list to record number of assigned tasks to a PE
        # for every scheduling instance
        assigned = [0]*(len(self.PEs))

        # go over all ready tasks for scheduling and make a decision
        for task in list_of_ready:

            exec_times = [np.inf]*(len(self.PEs))                                             # Initialize a list to keep execution times of task for each PE

            for i in range(len(self.resource_matrix.list)):
                if self.PEs[i].enabled:
                    if (task.name in self.resource_matrix.list[i].supported_functionalities):

                        ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)
                        exec_times[i] = self.resource_matrix.list[i].performance[ind]

            min_of_exec_times = min(exec_times)                                                 # $min_of_exec_times is the minimum of execution time of the task among all PEs
            count_minimum = exec_times.count(min_of_exec_times)                                 # also, record how many times $min_of_exec_times is seen in the list
            #print(count_minimum)

            # if there are two or more PEs satisfying minimum execution
            # then we should try to utilize all those PEs
            if (count_minimum > 1):

                # if there are tow or more PEs satisfying minimum execution
                # populate the IDs of those PEs into a list
                min_PE_IDs = [i for i, x in enumerate(exec_times) if x == min_of_exec_times]

                # then check whether those PEs are busy or idle
                PE_check_list = [True if not self.PEs[index].idle else False for i, index in enumerate(min_PE_IDs)]

                # assign tasks to the idle PEs instead of the ones that are currently busy
                if (True in PE_check_list) and (False in PE_check_list):
                    for PE in PE_check_list:
                        # if a PE is currently busy remove that PE from $min_PE_IDs list
                        # to schedule the task to a idle PE
                        if (PE == True):
                            min_PE_IDs.remove(min_PE_IDs[PE_check_list.index(PE)])

                # then compare the number of the assigned tasks to remaining PEs
                # and choose the one with the lowest number of assigned tasks
                assigned_tasks = [assigned[x] for i, x in enumerate(min_PE_IDs)]
                PE_ID_index = assigned_tasks.index(min(assigned_tasks))

                # finally, choose the best available PE for the task
                task.PE_ID = min_PE_IDs[PE_ID_index]

# =============================================================================
#                 # assign tasks to the idle PEs instead of the ones that are currently busy
#                 if (True in PE_check_list) and (False in PE_check_list):
#                     for PE in PE_check_list:
#                         # if a PE is currently busy remove that PE from $min_PE_IDs list
#                         # to schedule the task to a idle PE
#                         if (PE == True):
#                             min_PE_IDs.remove(min_PE_IDs[PE_check_list.index(PE)])
#
#
#                 # then compare the number of the assigned tasks to remaining PEs
#                 # and choose the one with the lowest number of assigned tasks
#                 assigned_tasks = [assigned[x] for i, x in enumerate(min_PE_IDs)]
#                 PE_ID_index = assigned_tasks.index(min(assigned_tasks))
# =============================================================================


                # finally, choose the best available PE for the task
                task.PE_ID = min_PE_IDs[PE_ID_index]

            else:
                task.PE_ID = exec_times.index(min_of_exec_times)
            # end of if count_minimum >1:
            # since one task is just assigned to a PE, increase the number by 1
            assigned[task.PE_ID] += 1

            if (task.PE_ID == -1):
                print ('[E] Time %s: %s can not be assigned to any resource, please check SoC.**.txt file'
                       % (self.env.now,task.name))
                print ('[E] or job_**.txt file')
                assert(task.PE_ID >= 0)
            else:
                if (common.INFO_SCH):
                    print ('[I] Time %s: The scheduler assigns the %s task to resource PE-%s: %s'
                           %(self.env.now, task.ID, task.PE_ID,
                             self.resource_matrix.list[task.PE_ID].type))
            # end of if task.PE_ID == -1:
        # end of for task in list_of_ready:
        # At the end of this loop, we should have a valid (non-negative ID)
        # that can run next_task

    # end of MET(list_of_ready)

    def EFT(self, list_of_ready):
        '''!
        This scheduler compares the execution times of the current task for available resources and also considers if a resource has
        already a task running. it picks the resource which will give the earliest finish time for the task
        @param list_of_ready: The list of ready tasks
        '''

        for task in list_of_ready:

            comparison = [np.inf]*len(self.PEs)                                     # Initialize the comparison vector
            comm_ready = [0]*len(self.PEs)                                          # A list to store the max communication times for each PE

            if (common.DEBUG_SCH):
                print ('[D] Time %s: The scheduler function is called with task %s'
                       %(self.env.now, task.ID))

            for i in range(len(self.resource_matrix.list)):
                if self.PEs[i].enabled:
                    # if the task is supported by the resource, retrieve the index of the task
                    if (task.name in self.resource_matrix.list[i].supported_functionalities):
                        ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)


                        # $PE_comm_wait_times is a list to store the estimated communication time
                        # (or the remaining communication time) of all predecessors of a task for a PE
                        # As simulation forwards, relevant data is being sent after a task is completed
                        # based on the time instance, one should consider either whole communication
                        # time or the remaining communication time for scheduling
                        PE_comm_wait_times = []

                        # $PE_wait_time is a list to store the estimated wait times for a PE
                        # till that PE is available if the PE is currently running a task
                        PE_wait_time = []

                        job_ID = -1                                                     # Initialize the job ID

                        # Retrieve the job ID which the current task belongs to
                        for ii, job in enumerate(self.jobs.list):
                            if job.name == task.jobname:
                                job_ID = ii

                        for predecessor in self.jobs.list[job_ID].task_list[task.base_ID].predecessors:
                            # data required from the predecessor for $ready_task
                            c_vol = self.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]

                            # retrieve the real ID  of the predecessor based on the job ID
                            real_predecessor_ID = predecessor + task.ID - task.base_ID

                            # Initialize following two variables which will be used if
                            # PE to PE communication is utilized
                            predecessor_PE_ID = -1
                            predecessor_finish_time = -1


                            for completed in common.TaskQueues.completed.list:
                                if completed.ID == real_predecessor_ID:
                                    predecessor_PE_ID = completed.PE_ID
                                    predecessor_finish_time = completed.finish_time
                                    #print(predecessor, predecessor_finish_time, predecessor_PE_ID)

                            if (common.PE_to_PE):
                                # Compute the PE to PE communication time
                                PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, i]
                                PE_to_PE_comm_time = int(c_vol/PE_to_PE_band)

                                if common.comm_model == 'PE' :
                                    PE_to_PE_comm_time = int(comm_vol/comm_band)
                                elif common.comm_model == 'NoC' :
                                    ## Compute communication time
                                    ## If source and destination are same PE, then communication time is assumed to be zero
                                    ## If not, compute the time as the number of Manhattan hops on the 2D-mesh
                                    pred_cluster = self.resource_matrix.list[predecessor_PE_ID].cluster_ID
                                    task_cluster = self.resource_matrix.list[task.PE_ID].cluster_ID

                                    if pred_cluster == task_cluster :
                                        PE_to_PE_comm_time = 0
                                    else :
                                        ## Get source and destination mesh positions and x,y indices
                                        src_mesh_pos = self.resource_matrix.list[predecessor_PE_ID].position
                                        dst_mesh_pos = self.resource_matrix.list[task.PE_ID].position

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

                                        PE_to_PE_comm_time = num_hops + (task.input_packet_size - 1)
                                ## if common.comm_model == 'PE' :


                                PE_comm_wait_times.append(max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))

                                if (common.DEBUG_SCH):
                                    print('[D] Time %s: Estimated communication time between PE %s to PE %s from task %s to task %s is %d'
                                          %(self.env.now, predecessor_PE_ID, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))

                            if (common.shared_memory):
                                # Compute the communication time considering the shared memory
                                # only consider memory to PE communication time
                                # since the task passed the 1st phase (PE to memory communication)
                                # and its status changed to ready

                                #PE_to_memory_band = common.ResourceManager.comm_band[predecessor_PE_ID, -1]
                                memory_to_PE_band = common.ResourceManager.comm_band[self.resource_matrix.list[-1].ID, i]
                                shared_memory_comm_time = int(c_vol/memory_to_PE_band)

                                PE_comm_wait_times.append(shared_memory_comm_time)
                                if (common.DEBUG_SCH):
                                    print('[D] Time %s: Estimated communication time between memory to PE %s from task %s to task %s is %d'
                                          %(self.env.now, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))

                            # $comm_ready contains the estimated communication time
                            # for the resource in consideration for scheduling
                            # maximum value is chosen since it represents the time required for all
                            # data becomes available for the resource.
                            comm_ready[i] = (max(PE_comm_wait_times))
                        # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors:

                        # if a resource currently is executing a task, then the estimated remaining time
                        # for the task completion should be considered during scheduling
                        PE_wait_time.append(max((self.PEs[i].available_time - self.env.now), 0))

                        reconfig_latency = DASH_acc_utils.get_DAP_reconfiguration_overhead(self.PEs[i], self.resource_matrix.list[self.PEs[i].ID], task)

                        # update the comparison vector accordingly
                        comparison[i] = self.resource_matrix.list[i].performance[ind] + max(max(comm_ready[i], PE_wait_time[-1]), reconfig_latency)
                    # end of if (task.name in...
            # end of for i in range(len(self.resource_matrix.list)):

            # after going over each resource, choose the one which gives the minimum result
            task.PE_ID = comparison.index(min(comparison))

            if task.PE_ID == -1:
                print ('[E] Time %s: %s can not be assigned to any resource, please check SoC.**.txt file'
                       % (self.env.now,task.ID))
                print ('[E] or job_**.txt file')
                assert(task.PE_ID >= 0)
            else:
                if (common.DEBUG_SCH):
                    print('[D] Time %s: Estimated execution times for each PE with task %s, respectively'
                              %(self.env.now, task.ID))
                    print('%12s'%(''), comparison)
                    print ('[D] Time %s: The scheduler assigns task %s to resource %s: %s'
                           %(self.env.now, task.ID, task.PE_ID, self.resource_matrix.list[task.PE_ID].type))

            # Finally, update the estimated available time of the resource to which
            # a task is just assigned
            self.PEs[task.PE_ID].available_time = self.env.now + comparison[task.PE_ID] + common.scheduling_overhead
            task.scheduling_overhead = common.scheduling_overhead
            # At the end of this loop, we should have a valid (non-negative ID)
            # that can run next_task

        # end of for task in list_of_ready:

    #end of EFT(list_of_ready)

    def STF(self, list_of_ready):
        '''!
        This scheduler compares the execution times of the current task for available resources and returns the ID of the resource
        with minimum execution time for the current task. The only difference between STF and MET is the order in which the tasks 
        are scheduled onto resources
        @param list_of_ready: The list of ready tasks
        '''

        ready_list = copy.deepcopy(list_of_ready)

        # Iterate through the list of ready tasks until all of them are scheduled
        while (len(ready_list) > 0) :

            shortest_task_exec_time = np.inf
            shortest_task_pe_id     = -1

            for task in ready_list:

                min_time = np.inf                                                                   # Initialize the best performance found so far as a large number

                for i in range(len(self.resource_matrix.list)):
                    if self.PEs[i].enabled:
                        if (task.name in self.resource_matrix.list[i].supported_functionalities):
                            ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)


                            if (self.resource_matrix.list[i].performance[ind] < min_time):              # Found resource with smaller execution time
                                min_time    = self.resource_matrix.list[i].performance[ind]                # Update the best time found so far
                                resource_id = self.resource_matrix.list[i].ID                           # Record the ID of the resource
                                #task.PE_ID = i                                                          # Record the corresponding resource

                #print('[INFO] Task - %d, Resource - %d, Time - %d' %(task.ID, resource_id, min_time))
                # Obtain the ID and resource for the shortest task in the current iteration
                if (min_time < shortest_task_exec_time) :
                    shortest_task_exec_time = min_time
                    shortest_task_pe_id     = resource_id
                    shortest_task           = task
                # end of if (min_time < shortest_task_exec_time)

            # end of for task in list_of_ready:
            # At the end of this loop, we should have the minimum execution time
            # of a task across all resources

            # Assign PE ID of the shortest task
            index = [i for i,x in enumerate(list_of_ready) if x.ID == shortest_task.ID][0]
            list_of_ready[index].PE_ID = shortest_task_pe_id
            shortest_task.PE_ID        = shortest_task_pe_id

            if (common.DEBUG_SCH):
                print ('[I] Time %s: The scheduler function found task %d to be shortest on resource %d with %.1f'
                       %(self.env.now, shortest_task.ID, shortest_task.PE_ID, shortest_task_exec_time))

            if list_of_ready[index].PE_ID == -1:
                print ('[E] Time %s: %s can not be assigned to any resource, please check SoC.**.txt file'
                       % (self.env.now,shortest_task.name))
                print ('[E] or job_**.txt file')
                assert(shortest_task.PE_ID >= 0)
            else:
                if (common.INFO_SCH):
                    print ('[I] Time %s: The scheduler assigns the %s task to resource PE-%s: %s'
                           %(self.env.now, shortest_task.ID, shortest_task.PE_ID,
                             self.resource_matrix.list[shortest_task.PE_ID].type))
            # end of if shortest_task.PE_ID == -1:

            # Remove the task which got a schedule successfully
            for i, task in enumerate(ready_list) :
                if task.ID == shortest_task.ID :
                    ready_list.remove(task)

        # end of for task in list_of_ready:
        # At the end of this loop, all ready tasks are assigned to the resources
        # on which the execution times are minimum. The tasks will execute
        # in the order of increasing execution times


    # end of STF(list_of_ready)

    def ETF_LB(self, list_of_ready):
        '''!
        This scheduler compares the execution times of the current task for available resources and also considers if a resource has
        already a task running. it picks the resource which will give the earliest finish time for the task. Additionally, the task 
        with the lowest earliest finish time is scheduled first
        @param list_of_ready: The list of ready tasks
        '''

        ready_list = copy.deepcopy(list_of_ready)

        task_counter = 0
        assigned = self.assigned

        # Iterate through the list of ready tasks until all of them are scheduled
        while len(ready_list) > 0:

            shortest_task_exec_time = np.inf
            shortest_task_pe_id = -1
            shortest_comparison = [np.inf] * len(self.PEs)

            for task in ready_list:

                comparison = [np.inf] * len(self.PEs)  # Initialize the comparison vector
                comm_ready = [0] * len(self.PEs)  # A list to store the max communication times for each PE

                if (common.DEBUG_SCH):
                    print('[D] Time %s: The scheduler function is called with task %s'
                          % (self.env.now, task.ID))

                for i in range(len(self.resource_matrix.list)):
                    if self.PEs[i].enabled:
                        # if the task is supported by the resource, retrieve the index of the task
                        if (task.name in self.resource_matrix.list[i].supported_functionalities):
                            ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)

                            # $PE_comm_wait_times is a list to store the estimated communication time
                            # (or the remaining communication time) of all predecessors of a task for a PE
                            # As simulation forwards, relevant data is being sent after a task is completed
                            # based on the time instance, one should consider either whole communication
                            # time or the remaining communication time for scheduling
                            PE_comm_wait_times = []

                            # $PE_wait_time is a list to store the estimated wait times for a PE
                            # till that PE is available if the PE is currently running a task
                            PE_wait_time = []

                            job_ID = -1  # Initialize the job ID

                            # Retrieve the job ID which the current task belongs to
                            for ii, job in enumerate(self.jobs.list):
                                if job.name == task.jobname:
                                    job_ID = ii

                            for predecessor in self.jobs.list[job_ID].task_list[task.base_ID].predecessors:
                                # data required from the predecessor for $ready_task
                                c_vol = self.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]

                                # retrieve the real ID  of the predecessor based on the job ID
                                real_predecessor_ID = predecessor + task.ID - task.base_ID

                                # Initialize following two variables which will be used if
                                # PE to PE communication is utilized
                                predecessor_PE_ID = -1
                                predecessor_finish_time = -1

                                for completed in common.TaskQueues.completed.list:
                                    if completed.ID == real_predecessor_ID:
                                        predecessor_PE_ID = completed.PE_ID
                                        predecessor_finish_time = completed.finish_time
                                        # print(predecessor, predecessor_finish_time, predecessor_PE_ID)
                                        break

                                if (common.PE_to_PE):
                                    # Compute the PE to PE communication time
                                    PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, i]
                                    PE_to_PE_comm_time = int(c_vol / PE_to_PE_band)

                                    PE_comm_wait_times.append(max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))

                                    if (common.DEBUG_SCH):
                                        print('[D] Time %s: Estimated communication time between PE-%s to PE-%s from task %s to task %s is %d'
                                              % (self.env.now, predecessor_PE_ID, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))

                                if (common.shared_memory):
                                    # Compute the communication time considering the shared memory
                                    # only consider memory to PE communication time
                                    # since the task passed the 1st phase (PE to memory communication)
                                    # and its status changed to ready

                                    # PE_to_memory_band = common.ResourceManager.comm_band[predecessor_PE_ID, -1]
                                    memory_to_PE_band = common.ResourceManager.comm_band[self.resource_matrix.list[-1].ID, i]
                                    shared_memory_comm_time = int(c_vol / memory_to_PE_band)

                                    PE_comm_wait_times.append(shared_memory_comm_time)
                                    if (common.DEBUG_SCH):
                                        print('[D] Time %s: Estimated communication time between memory to PE-%s from task %s to task %s is %d'
                                              % (self.env.now, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))

                                # $comm_ready contains the estimated communication time
                                # for the resource in consideration for scheduling
                                # maximum value is chosen since it represents the time required for all
                                # data becomes available for the resource.
                                comm_ready[i] = max(PE_comm_wait_times)
                            # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors:

                            # if a resource currently is executing a task, then the estimated remaining time
                            # for the task completion should be considered during scheduling
                            PE_wait_time.append(max((self.PEs[i].available_time - self.env.now), 0))

                            reconfig_latency = DASH_acc_utils.get_DAP_reconfiguration_overhead(self.PEs[i], self.resource_matrix.list[self.PEs[i].ID], task)

                            # update the comparison vector accordingly
                            comparison[i] = self.resource_matrix.list[i].performance[ind] * (1 + DTPM_power_models.compute_DVFS_performance_slowdown(common.ClusterManager.cluster_list[self.PEs[i].cluster_ID])) + max(max(comm_ready[i], PE_wait_time[-1]), reconfig_latency)
                        # end of if (task.name in...
                # end of for i in range(len(self.resource_matrix.list)):

                if min(comparison) < shortest_task_exec_time:
                    resource_id = comparison.index(min(comparison))
                    shortest_task_exec_time = min(comparison)
                    #                    print(shortest_task_exec_time, comparison)
                    count_minimum = comparison.count(shortest_task_exec_time)  # also, record how many times $min_of_exec_times is seen in the list
                    # if there are two or more PEs satisfying minimum execution
                    # then we should try to utilize all those PEs
                    if (count_minimum > 1):
                        # if there are two or more PEs satisfying minimum execution
                        # populate the IDs of those PEs into a list
                        min_PE_IDs = [i for i, x in enumerate(comparison) if x == shortest_task_exec_time]
                        # then compare the number of the assigned tasks to remaining PEs
                        # and choose the one with the lowest number of assigned tasks
                        assigned_tasks = [assigned[x] for i, x in enumerate(min_PE_IDs)]
                        PE_ID_index = assigned_tasks.index(min(assigned_tasks))

                        # finally, choose the best available PE for the task
                        task.PE_ID = min_PE_IDs[PE_ID_index]
                    #   print(count_minimum, task.PE_ID)
                    else:
                        task.PE_ID = comparison.index(shortest_task_exec_time)
                    # end of if count_minimum >1:

                    # since one task is just assigned to a PE, increase the number by 1
                    assigned[task.PE_ID] += 1

                    resource_id = task.PE_ID
                    shortest_task_pe_id = resource_id
                    shortest_task = task
                    shortest_comparison = copy.deepcopy(comparison)

            # assign PE ID of the shortest task
            index = [i for i, x in enumerate(list_of_ready) if x.ID == shortest_task.ID][0]
            list_of_ready[index].PE_ID = shortest_task_pe_id
            list_of_ready[index], list_of_ready[task_counter] = list_of_ready[task_counter], list_of_ready[index]
            shortest_task.PE_ID = shortest_task_pe_id

            if shortest_task.PE_ID == -1:
                print('[E] Time %s: %s can not be assigned to any resource, please check SoC.**.txt file'
                      % (self.env.now, shortest_task.ID))
                print('[E] or job_**.txt file')
                assert (task.PE_ID >= 0)
            else:
                if (common.DEBUG_SCH):
                    print('[D] Time %s: Estimated execution times for each PE with task %s, respectively'
                          % (self.env.now, shortest_task.ID))
                    print('%12s' % (''), comparison)
                    print('[D] Time %s: The scheduler assigns task %s to PE-%s: %s'
                          % (self.env.now, shortest_task.ID, shortest_task.PE_ID, self.resource_matrix.list[shortest_task.PE_ID].name))

            # Finally, update the estimated available time of the resource to which
            # a task is just assigned
            index_min_available_time = self.PEs[shortest_task.PE_ID].available_time_list.index(min(self.PEs[shortest_task.PE_ID].available_time_list))
            self.PEs[shortest_task.PE_ID].available_time_list[index_min_available_time] = self.env.now + shortest_comparison[shortest_task.PE_ID] + common.scheduling_overhead
            shortest_task.scheduling_overhead = common.scheduling_overhead
            self.PEs[shortest_task.PE_ID].available_time = min(self.PEs[shortest_task.PE_ID].available_time_list)

            # Remove the task which got a schedule successfully
            for i, task in enumerate(ready_list):
                if task.ID == shortest_task.ID:
                    ready_list.remove(task)

            task_counter += 1
            # At the end of this loop, we should have a valid (non-negative ID)
            # that can run next_task

        # end of while len(ready_list) > 0 :

    # end of ETF( list_of_ready)

    def HEFT(self, list_of_ready):
        '''!
        Schedule using the HEFT heuristic. As HEFT is a static scheduler, we simply read off the lookup table that was generated during the last job generation epoch in job_generator.py
        @param list_of_ready: the list of ready tasks that need to be scheduled
        '''
        for task in list_of_ready:
            task.PE_ID = common.table[task.ID][0]
            task.order = common.table[task.ID][1]
            task.dynamic_dependencies = common.table[task.ID][2]
        list_of_ready.sort(key=lambda task: task.order)
    #end of HEFT(self, list_of_ready)

    def HEFT_RT(self, list_of_ready):
        '''!
        Schedule with the HEFT_RT heuristic. Rather than use a static lookup table, HEFT_RT considers the impact of scheduling a dag consisting of just the tasks in the ready queue.
        This DAG is then passed through the full HEFT algorithm on each scheduling epoch.
        @param list_of_ready: the list of ready tasks that need to be scheduled
        '''
        if not list_of_ready:
            return
        computation_dict = {}
        power_dict = {}
        dag = nx.DiGraph()
        for task in list_of_ready:
            dag.add_node(task.ID)
            computation_dict[task.ID] = []
            power_dict[task.ID] = []
            for cluster in common.ClusterManager.cluster_list:
                current_power = cluster.current_power_cluster
                for resource_idx in cluster.PE_list:
                    resource = self.resource_matrix.list[resource_idx]
                    if task.name in resource.supported_functionalities:
                        perf_index = resource.supported_functionalities.index(task.name)
                        computation_dict[task.ID].append(resource.performance[perf_index])
                        power_dict[task.ID].append(current_power / len(cluster.PE_list))
                    else:
                        computation_dict[task.ID].append(np.inf)
                        power_dict[task.ID].append(np.inf)


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

        dag = dag_merge.merge_dags(dag, merge_method=merge_method, skip_relabeling=True)
        computation_dict[max(dag) - 1] = np.zeros((1, len(self.resource_matrix.list)))
        computation_dict[max(dag)] = np.zeros((1, len(self.resource_matrix.list)))
        power_dict[max(dag) - 1] = np.zeros((1, len(self.resource_matrix.list)))
        power_dict[max(dag)] = np.zeros((1, len(self.resource_matrix.list)))
        computation_matrix = np.empty((max(dag) + 1, len(self.resource_matrix.list)))
        power_matrix = np.empty((max(dag) + 1, len(self.resource_matrix.list)))

        running_tasks = {}
        for idx in range(len(self.resource_matrix.list)):
            running_tasks[idx] = []

        for task in common.TaskQueues.running.list:
            executing_resource = self.resource_matrix.list[task.PE_ID]
            task_id = task.ID
            task_start = task.start_time
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heft.ScheduleEvent(task_id, task_start, task_end, proc))

        for task in common.TaskQueues.executable.list:
            executing_resource = self.resource_matrix.list[task.PE_ID]
            task_id = task.ID
            if len(running_tasks[task.PE_ID]) != 0:
                task_start = running_tasks[task.PE_ID][-1].end
            else:
                task_start = self.env.now
            task_end = task_start + executing_resource.performance[executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heft.ScheduleEvent(task_id, task_start, task_end, proc))

        for key, val in computation_dict.items():
            computation_matrix[key, :] = val
        for key, val in power_dict.items():
            power_matrix[key, :] = val
        _, _, dict_output = heft.schedule_dag(
            dag,
            computation_matrix=computation_matrix,
            communication_matrix=common.ResourceManager.comm_band,
            time_offset=self.env.now,
            proc_schedules=running_tasks,
            relabel_nodes=False,
            rank_metric=rank_metric,
            power_dict=power_matrix,
            op_mode=op_mode
        )
        for task in list_of_ready:
            task.PE_ID = dict_output[task.ID][0]
            task.dynamic_dependencies = dict_output[task.ID][2]
    # end of HEFT_RT(self, list_of_ready)

    def PEFT(self, list_of_ready):
        '''!
        Schedule using the PEFT heuristic. As PEFT is a static scheduler, we simply read off the lookup table that was generated during the last job generation epoch in job_generator.py
        @param list_of_ready: the list of ready tasks that need to be scheduled
        '''
        for task in list_of_ready:
            task.PE_ID = common.table[task.ID][0]
            task.order = common.table[task.ID][1]
            task.dynamic_dependencies = common.table[task.ID][2]
        list_of_ready.sort(key=lambda task: task.order)
    # end of PEFT(self, list_of_ready)

    def PEFT_RT(self, list_of_ready):
        '''!
        Schedule with the PEFT_RT heuristic. Rather than use a static lookup table, PEFT_RT considers the impact of scheduling a dag consisting of just the tasks in the ready queue.
        This DAG is then passed through the full PEFT algorithm on each scheduling epoch.
        @param list_of_ready: the list of ready tasks that need to be scheduled
        '''
        if not list_of_ready:
            return
        computation_dict = {}
        dag = nx.DiGraph()
        for task in list_of_ready:
            dag.add_node(task.ID)
            computation_dict[task.ID] = []
            for cluster in common.ClusterManager.cluster_list:
                for resource_idx in cluster.PE_list:
                    resource = self.resource_matrix.list[resource_idx]
                    if task.name in resource.supported_functionalities:
                        perf_index = resource.supported_functionalities.index(task.name)
                        computation_dict[task.ID].append(resource.performance[perf_index])
                    else:
                        computation_dict[task.ID].append(np.inf)
        dag = dag_merge.merge_dags(dag, merge_method=dag_merge.MergeMethod.COMMON_ENTRY_EXIT, skip_relabeling=True)
        computation_dict[max(dag) - 1] = np.zeros((1, len(self.resource_matrix.list)))
        computation_dict[max(dag)] = np.zeros((1, len(self.resource_matrix.list)))
        computation_matrix = np.empty((max(dag) + 1, len(self.resource_matrix.list)))

        running_tasks = {}
        for idx in range(len(self.resource_matrix.list)):
            running_tasks[idx] = []

        for task in common.TaskQueues.running.list:
            executing_resource = self.resource_matrix.list[task.PE_ID]
            task_id = task.ID
            task_start = task.start_time
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heft.ScheduleEvent(task_id, task_start, task_end, proc))

        for task in common.TaskQueues.executable.list:
            executing_resource = self.resource_matrix.list[task.PE_ID]
            task_id = task.ID
            if len(running_tasks[task.PE_ID]) != 0:
                task_start = running_tasks[task.PE_ID][-1].end
            else:
                task_start = self.env.now
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heft.ScheduleEvent(task_id, task_start, task_end, proc))

        for key, val in computation_dict.items():
            computation_matrix[key, :] = val
        _, _, dict_output = peft.schedule_dag(
            dag,
            computation_matrix=computation_matrix,
            communication_matrix=common.ResourceManager.comm_band,
            time_offset=self.env.now,
            proc_schedules=running_tasks,
            relabel_nodes=False
        )
        for task in list_of_ready:
            task.PE_ID = dict_output[task.ID][0]
            task.dynamic_dependencies = dict_output[task.ID][2]
    # end of PEFT_RT(self, list_of_ready)

    
    def ETF(self, list_of_ready):
        '''
        This scheduler compares the execution times of the current
        task for available resources and also considers if a resource has
        already a task running. it picks the resource which will give the
        earliest finish time for the task. Additionally, the task with the
        lowest earliest finish time  is scheduled first
        '''
        ready_list = copy.deepcopy(list_of_ready)
    
        ##################################################################################################
        ## IL-Scheduler
        ##################################################################################################

        if common.ils_enable_dataset_save or common.ils_enable_policy_decision :

            ## Open file handles
            scale_values = str(common.scale_values_list[0]) + '-' + str(common.scale_values_list[0] + 1) + '-1'
            
            ## Set filenames
            ils_clustera_filename = './datasets/data_IL_clustera_' + scale_values + common.dagger_string + '.csv'

            ## Obtain number of clusters and number of PEs in each cluster
            num_PE_list = []
            num_clusters = 0
            for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :
                ## Skip processing for memory
                if cluster.name == 'MEMORY' :
                    continue
                ## if cluster.name == 'MEMORY' :

                num_clusters += 1
                num_PE_list.append(cluster.num_total_cores)
            ## for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :
            max_num_PEs = np.max(num_PE_list)

            ## Print file headers for the first time
            if common.ils_filename_header == 0 :
               if common.ils_classifier_type == 'NN' :
                   from keras.models import load_model
               ## if common.ils_classifier_type == 'NN' :
               ils_scripts.ils_print_file_headers('Cluster', ils_clustera_filename, num_clusters, num_clusters) 
            ## if common.ils_filename_header == 1 :

            ils_clustera_fp = open(ils_clustera_filename, 'a')

            ## Create file handles for each cluster
            for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :
                ## Skip processing for memory
                if cluster.name == 'MEMORY' :
                    continue
                ## if cluster.name == 'MEMORY' :

                vars()['ils_cluster' + str(cluster_index) + '_filename'] = './datasets/data_IL_cluster' + str(cluster_index) + '_' + scale_values + common.dagger_string + '.csv'
                ## Print file headers for the first time
                if common.ils_filename_header == 0 :
                   ## ils_scripts.ils_print_file_headers('PE', vars()['ils_cluster' + str(cluster_index) + '_filename'], num_clusters, num_PE_list[cluster_index])
                   ils_scripts.ils_print_file_headers('PE', vars()['ils_cluster' + str(cluster_index) + '_filename'], num_clusters, max_num_PEs)
                ## if common.ils_filename_header == 1 :

                vars()['ils_cluster' + str(cluster_index) + '_fp'] = open(vars()['ils_cluster' + str(cluster_index) + '_filename'], 'a') 
            ## for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :
            
            common.ils_filename_header = 1

        ## if common.ils_enable_dataset_save or common.ils_enable_policy_decision :
        
        if common.ils_enable_dataset_save or common.ils_enable_policy_decision :
            ## Get minimum and maximum job ID of ready tasks
            job_id_list = []
            for task in ready_list :
                if task.jobID not in job_id_list :
                    job_id_list.append(task.jobID)
                ## if task.jobID not in job_id_list :
            ## for task in list_of_ready :
            job_id_list = np.array(job_id_list)
            min_job_id = np.min(job_id_list)
            max_job_id = np.max(job_id_list)
        ## if common.ils_enable_dataset_save or common.ils_enable_policy_decision :

        ##################################################################################################
        ## End of IL-Scheduler
        ##################################################################################################

        # Iterate through the list of ready tasks until all of them are scheduled
        task_index = 0
        while len(ready_list) > 0 :
    
            shortest_task_exec_time = np.inf
            shortest_task_pe_id     = -1
            shortest_comparison     = [np.inf] * len(self.PEs)
            
            ## Choose task from list_of_ready
            if common.ils_enable_policy_decision :
                shortest_task = list_of_ready[task_index]
            ## if common.ils_enable_policy_decision :
    
            ## Compute task with lowest execution time and corresponding PE
            ## Compute only if IL-scheduler policy is not enabled
            if not common.ils_enable_policy_decision :
                for task in ready_list:
                    
                    comparison = [np.inf]*len(self.PEs)                                     # Initialize the comparison vector 
                    comm_ready = [0]*len(self.PEs)                                          # A list to store the max communication times for each PE
                    
                    if (common.DEBUG_SCH):
                        print ('[D] Time %s: The scheduler function is called with task %s'
                               %(self.env.now, task.ID))
                        
                    for i in range(len(self.resource_matrix.list)):
                        # if the task is supported by the resource, retrieve the index of the task
                        if (task.name in self.resource_matrix.list[i].supported_functionalities):
                            ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)
                            
                                
                            # $PE_comm_wait_times is a list to store the estimated communication time 
                            # (or the remaining communication time) of all predecessors of a task for a PE
                            # As simulation forwards, relevant data is being sent after a task is completed
                            # based on the time instance, one should consider either whole communication
                            # time or the remaining communication time for scheduling
                            PE_comm_wait_times = []
                            
                            # $PE_wait_time is a list to store the estimated wait times for a PE
                            # till that PE is available if the PE is currently running a task
                            PE_wait_time = []
                              
                            job_ID = -1                                                     # Initialize the job ID
                            
                            # Retrieve the job ID which the current task belongs to
                            for ii, job in enumerate(self.jobs.list):
                                if job.name == task.jobname:
                                    job_ID = ii
                                    
                            for predecessor in self.jobs.list[job_ID].task_list[task.base_ID].predecessors:
                                # data required from the predecessor for $ready_task
                                c_vol = self.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]
                                
                                # retrieve the real ID  of the predecessor based on the job ID
                                real_predecessor_ID = predecessor + task.ID - task.base_ID
                                
                                # Initialize following two variables which will be used if 
                                # PE to PE communication is utilized
                                predecessor_PE_ID = -1
                                predecessor_finish_time = -1
                                
                                
                                for completed in common.TaskQueues.completed.list:
                                    if (completed.ID == real_predecessor_ID):
                                        predecessor_PE_ID = completed.PE_ID
                                        predecessor_finish_time = completed.finish_time
                                        #print(predecessor, predecessor_finish_time, predecessor_PE_ID)
                                        
                                
                                if (common.PE_to_PE):
                                    # Compute the PE to PE communication time
                                    #PE_to_PE_band = self.resource_matrix.comm_band[predecessor_PE_ID, i]
                                    PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, i]

                                    if common.comm_model == 'PE' :
                                        PE_to_PE_comm_time = int(c_vol / PE_to_PE_band)
                                    elif common.comm_model == 'NoC' :
                                        ## Compute communication time
                                        ## If source and destination are same PE, then communication time is assumed to be zero
                                        ## If not, compute the time as the number of Manhattan hops on the 2D-mesh
                                        pred_cluster = self.resource_matrix.list[predecessor_PE_ID].cluster_ID
                                        task_cluster = self.resource_matrix.list[task.PE_ID].cluster_ID

                                        if pred_cluster == task_cluster :
                                            PE_to_PE_comm_time = 0
                                        else :
                                            ## Get source and destination mesh positions and x,y indices
                                            src_mesh_pos = self.resource_matrix.list[predecessor_PE_ID].position
                                            dst_mesh_pos = self.resource_matrix.list[task.PE_ID].position

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

                                            PE_to_PE_comm_time = num_hops + (task.input_packet_size - 1)
                                    ## if common.comm_model == 'PE' :
                                    
                                    PE_comm_wait_times.append(max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))
                                    
                                    if (common.DEBUG_SCH):
                                        print('[D] Time %s: Estimated communication time between PE-%s to PE-%s from task %s to task %s is %d' 
                                              %(self.env.now, predecessor_PE_ID, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))
                                    
                                if (common.shared_memory):
                                    # Compute the communication time considering the shared memory
                                    # only consider memory to PE communication time
                                    # since the task passed the 1st phase (PE to memory communication)
                                    # and its status changed to ready 
                                    
                                    #PE_to_memory_band = self.resource_matrix.comm_band[predecessor_PE_ID, -1]
                                    memory_to_PE_band = common.ResourceManager.comm_band[self.resource_matrix.list[-1].ID, i]
                                    shared_memory_comm_time = int(c_vol/memory_to_PE_band)
                                    
                                    PE_comm_wait_times.append(shared_memory_comm_time)
                                    if (common.DEBUG_SCH):
                                        print('[D] Time %s: Estimated communication time between memory to PE-%s from task %s to task %s is %d' 
                                              %(self.env.now, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))
                                
                                # $comm_ready contains the estimated communication time 
                                # for the resource in consideration for scheduling
                                # maximum value is chosen since it represents the time required for all
                                # data becomes available for the resource. 
                                comm_ready[i] = (max(PE_comm_wait_times))
                                
                            # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors: 
                            
                            # if a resource currently is executing a task, then the estimated remaining time
                            # for the task completion should be considered during scheduling
                            PE_wait_time.append(max((self.PEs[i].available_time - self.env.now), 0))

                            reconfig_latency = DASH_acc_utils.get_DAP_reconfiguration_overhead(self.PEs[i], self.resource_matrix.list[self.PEs[i].ID], task)

                            # update the comparison vector accordingly    
                            comparison[i] = self.resource_matrix.list[i].performance[ind] + max(max(comm_ready[i], PE_wait_time[-1]), reconfig_latency)
                            
    
                            # after going over each resource, choose the one which gives the minimum result
                            resource_id = comparison.index(min(comparison))
                            #print('aa',comparison)
                        # end of if (task.name in self.resource_matrix.list[i]...
                        
                    # obtain the task ID, resource for the task with earliest finish time 
                    # based on the computation 
                    #print('bb',comparison)
                    if min(comparison) < shortest_task_exec_time :
                        shortest_task_exec_time = min(comparison)
                        shortest_task_pe_id     = resource_id
                        shortest_task           = task
                        shortest_comparison     = comparison
                        expected_latency        = shortest_comparison[shortest_task_pe_id]
                        
                    # end of for i in range(len(self.resource_matrix.list)):
                # end of for task in ready_list:
                
                # assign PE ID of the shortest task 
                index = [i for i,x in enumerate(list_of_ready) if x.ID == shortest_task.ID][0]
                list_of_ready[index].PE_ID = shortest_task_pe_id
                list_of_ready[index], list_of_ready[task_index] = list_of_ready[task_index], list_of_ready[index]
                shortest_task.PE_ID        = shortest_task_pe_id
    
                if shortest_task.PE_ID == -1:
                    print ('[E] Time %s: %s can not be assigned to any resource, please check DASH.SoC.**.txt file'
                           % (self.env.now, shortest_task.ID))
                    print ('[E] or job_**.txt file')
                    assert(task.PE_ID >= 0)           
                else: 
                    if (common.DEBUG_SCH):
                        print('[D] Time %s: Estimated execution times for each PE with task %s, respectively' 
                                  %(self.env.now, shortest_task.ID))
                        print('%12s'%(''), comparison)
                        print ('[D] Time %s: The scheduler assigns task %s to PE-%s: %s'
                               %(self.env.now, shortest_task.ID, shortest_task.PE_ID, 
                                 self.resource_matrix.list[shortest_task.PE_ID].name))
            ## if not common.ils_enable_policy_decision :
            
            ##################################################################################################
            ## IL-Scheduler
            ##################################################################################################

            if common.ils_enable_dataset_save or common.ils_enable_policy_decision :

                ## Initialize a list to store all IL related features
                il_features     = []

                ##############################################################
                ## Cluster/PE availability time features
                ##############################################################

                ## Collect cluster available times
                cluster_free_times    = []
                cluster_PE_free_times = []

                ## Track max PEs within each cluster
                max_PEs_in_cluster = 0

                ## Iterate through all clusters to populate availability times
                for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :

                    ## Skip processing for memory
                    if cluster.name == 'MEMORY' :
                        continue
                    ## if cluster.name == 'MEMORY' :

                    ## Track max PEs within cluster
                    if cluster.num_total_cores > max_PEs_in_cluster :
                        max_PEs_in_cluster = cluster.num_total_cores
                    ## if cluster.num_total_cores > max_PEs_in_cluster :

                    ## Get free times of PEs within cluster
                    PE_free_times = []
                    for PE in self.PEs :
                        if PE.ID in cluster.PE_list :
                           ## The available time is divided by 30 for scaling
                           free_time = (max(PE.available_time - self.env.now, 0)) / 30
                           PE_free_times.append(free_time)
                        ## if PE.ID in cluster.PE_list :
                    ## for PE in self.PEs :
                    PE_free_times = np.pad(PE_free_times, (0, (max_num_PEs - len(PE_free_times))), constant_values=(50), mode='constant')

                    ## Populate free time of cluster as the earliest time a PE is available within the cluster
                    cluster_free_times.append(min(PE_free_times))
                    
                    ## Populate free time of PEs within cluster
                    cluster_PE_free_times.append(PE_free_times)
                    
                ## for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :

                ## Convert arrays to numpy arrays
                cluster_free_times    = np.array(cluster_free_times)
                cluster_PE_free_times = np.array(cluster_PE_free_times)

                ##############################################################
                ## Task execution time features
                ##############################################################

                ## Populate execution time profile (cluster-wise)
                task_exec_times = []
                for cluster in common.ClusterManager.cluster_list :
                    ## Skip processing for memory
                    if cluster.name == 'MEMORY' :
                        continue
                    ## if cluster.name == 'MEMORY' :

                    ## Obtain execution time of task for each cluster
                    ## To do so, obtain the execution time of any one PE (the 1st PE) within each cluster
                    PE_ID = cluster.PE_list[0]
                    if shortest_task.name in self.resource_matrix.list[PE_ID].supported_functionalities :
                        ind = self.resource_matrix.list[PE_ID].supported_functionalities.index(shortest_task.name)
                        exec_time = self.resource_matrix.list[PE_ID].performance[ind]
                        task_exec_times.append(exec_time)
                    else :
                        task_exec_times.append(10000)
                    ## if shortest_task.name in self.resource_matrix.list[PE.ID].supported_functionalities :
                ## for cluster in common.ClusterManager.cluster_list :

                ## Normalize cluster execution times
                task_exec_times = np.array(task_exec_times)
                # if len(task_exec_times[np.all([task_exec_times != 0, task_exec_times != 10000])]) == 0 :
                if len(task_exec_times[np.all([task_exec_times != 0])]) == 0 :
                    normalized_task_exec_times = np.array([0.0,0.0])
                    num_clusters = 0
                    for cluster in common.ClusterManager.cluster_list :
                        ## Skip processing for memory
                        if cluster.name == 'MEMORY' :
                            continue
                        ## if cluster.name == 'MEMORY' :
                        num_clusters += 1
                    ## for cluster in common.ClusterManager.cluster_list :
                    normalized_task_exec_times = np.pad(normalized_task_exec_times, (0, (num_clusters - 2)), constant_values=(10), mode='constant')
                else :
                    normalized_task_exec_times = ils_scripts.get_normalized_list(task_exec_times)
                ## if len(task_exec_times[np.all([task_exec_times != 0])]) == 0 :
                
                ##############################################################
                ## Task depth feature
                ##############################################################

                ## Populate normalized downward depth of task in DAG
                task_depth = shortest_task.dag_depth
                task_job_index = -1
                for job_index, job in enumerate(self.jobs.list) :
                    if shortest_task.jobname == job.name :
                        job_depth  = self.jobs.list[job_index].dag_depth['DAG']
                        num_tasks  = len(self.jobs.list[job_index].task_list)
                        task_job_index = job_index
                normalized_task_depth = task_depth / job_depth

                ##############################################################
                ## Relative job ID feature
                ##############################################################

                ## Populate relative job ID
                if max_job_id - min_job_id == 0 :
                    relative_job_id = 0
                else :
                    relative_job_id = (shortest_task.jobID - min_job_id) / (max_job_id - min_job_id)
                ## if max_job_id - min_job_id == 0 :

                ##############################################################
                ## Predecessor related dfeatures
                ##############################################################

                pred_task_comm_time_list = []
                predecessor_PEs = []

                ## Populate predecessor PE IDs
                task_preds = np.array(shortest_task.preds)
                task_pred_IDs = task_preds + shortest_task.head_ID
                task_preds_cluster_list = []
                task_preds_comm_vol = []
                for task_pred in task_pred_IDs :
                    for completed_task in common.TaskQueues.completed.list :
                        if completed_task.ID == task_pred :
                            task_preds_cluster_list.append(ils_scripts.get_cluster(completed_task.PE_ID))
                            c_vol = self.jobs.list[task_job_index].comm_vol[task_pred - shortest_task.head_ID, shortest_task.base_ID]
                            task_preds_comm_vol.append(c_vol / 1000)
                            ## for PE in self.PEs :
                        ## if completed_task.ID == task_pred :
                    ## for completed_task in common.TaskQueues.completed.list :
                ## for task_pred in task_preds :

                num_short_preds = 5 - len(shortest_task.preds)
                for num in range(num_short_preds) :
                    task_preds = np.append(task_preds, 10000)
                    task_preds_cluster_list.append(10000)
                    task_preds_comm_vol.append(10000)
                ## for num in range(num_short_preds) :

                task_preds_cluster_list = np.array(task_preds_cluster_list)
                task_preds_cluster_list = task_preds_cluster_list / 4
                task_preds_cluster_list[task_preds_cluster_list > 1000] = 10
                task_preds[task_preds > 1000] = 50
                task_preds_comm_vol = np.array(task_preds_comm_vol)
                task_preds_comm_vol[task_preds_comm_vol > 1000] = 10

                ##############################################################
                ## Few other features
                ##############################################################

                # Populate job type
                job_type = -1
                
                # Retrieve the job ID which the current task belongs to
                for job_index, job in enumerate(self.jobs.list):
                    if job.name == shortest_task.jobname:
                        job_type = job_index
                ## for job_index, job in enumerate(self.jobs.list):
                                
                # Populate normalized task ID
                normalized_task_ID = shortest_task.base_ID / num_tasks

                ##############################################################
                ## Populate all features into IL feature list
                ##############################################################

                # Populate features and save samples
                il_features.append(normalized_task_ID)
                il_features.extend(normalized_task_exec_times)
                il_features.append(normalized_task_depth)
                il_features.append(relative_job_id)
                il_features.append(job_type)
                il_features.extend(task_preds)
                il_features.extend(task_preds_cluster_list)
                il_features.extend(task_preds_comm_vol)

                #####################################################################
                ## Populate all data into respective files if dataset should be saved
                #####################################################################

                if common.ils_enable_dataset_save :
                    ## Print resource label
                    resource_label = shortest_task.PE_ID
                    cluster_label = ils_scripts.get_cluster(shortest_task.PE_ID)

                    ils_clustera_fp.write(str(self.env.now) + ',')
                    ils_clustera_fp.write(str(shortest_task.ID))
                    for feature in cluster_free_times :
                        ils_clustera_fp.write(',' + str(feature))
                    for feature in il_features :
                        ils_clustera_fp.write(',' + str(feature))
                    ils_clustera_fp.write(',' + str(resource_label))
                    ils_clustera_fp.write(',' + str(cluster_label))
                    ils_clustera_fp.write('\n')

                    file_handle = vars()['ils_cluster' + str(cluster_label) + '_fp']
                    file_handle.write(str(self.env.now) + ',')
                    file_handle.write(str(shortest_task.ID))
                    for feature in cluster_PE_free_times[cluster_label] :
                        file_handle.write(',' + str(feature))
                    for feature in il_features :
                        file_handle.write(',' + str(feature))
                    file_handle.write(',' + str(resource_label))
                    file_handle.write(',' + str(cluster_label))
                    file_handle.write('\n')
                ## if common.ils_enable_dataset_save :

                #####################################################################
                ## Use IL policy to make scheduling decisions
                #####################################################################
                
                if common.ils_enable_policy_decision :

                    ## Process features to feed to model
                    features_to_cluster_model = cluster_free_times
                    features_to_cluster_model = np.append(features_to_cluster_model, np.array(il_features)).reshape(1,-1)

                    ## Load cluster model
                    if common.ils_classifier_type == 'RT' :
                        il_clustera_model = pickle.load(open('./models/RT_clustera_merged' + common.model_dagger_string + '_model_' + str(common.ils_RT_tree_depth) + '.sav', 'rb'))
                    else :
                        il_clustera_model = load_model('./models/NN_clustera_merged' + common.model_dagger_string + '_model' + '.h5')
                    ## if common.ils_classifier_type == 'RT' :

                    ## Use policy to make scheduling decisions for cluster
                    if common.ils_classifier_type == 'NN' :
                        task_cluster_prediction = np.argmax(il_clustera_model.predict(features_to_cluster_model))
                    else :
                        task_cluster_prediction = int(np.around(il_clustera_model.predict(features_to_cluster_model)[0]))

                    ## Fail-safe mechanism
                    ## Get any one PE from cluster to check if task is supported by cluster
                    test_PE_index = common.ClusterManager.cluster_list[task_cluster_prediction].PE_list[0]

                    ## Check if task is supported by PE in the predicted cluster
                    if not shortest_task.name in  self.resource_matrix.list[test_PE_index].supported_functionalities :
                        task_cluster_prediction = 1
                    ## if not shortest_task.name in  self.resource_matrix.list[test_PE_index].supported_functionalities :

                    ## Process features to feed to model
                    features_to_PE_model = cluster_PE_free_times[task_cluster_prediction]
                    features_to_PE_model = np.append(features_to_PE_model, np.array(il_features)).reshape(1,-1)

                    ## Load PE model
                    if common.ils_classifier_type == 'RT' :
                        il_PE_model = pickle.load(open('./models/RT_cluster' + str(task_cluster_prediction) + '_merged' + common.model_dagger_string + '_model_' + str(common.ils_RT_tree_depth) + '.sav', 'rb'))
                    else :
                        il_PE_model = load_model('./models/NN_cluster' + str(task_cluster_prediction) + '_merged' + common.model_dagger_string + '_model' + '.h5')
                    ## if common.ils_classifier_type == 'RT' :

                    ## Use policy to make scheduling decisions for PE within clsuter
                    if common.ils_classifier_type == 'NN' :
                        task_PE_prediction = np.argmax(il_PE_model.predict(features_to_PE_model))
                    else :
                        task_PE_prediction = int(np.around(il_PE_model.predict(features_to_PE_model)[0]))
                    ## if common.ils_classifier_type == 'NN' :

                    ## Assign predicted PE to task
                    shortest_task.PE_ID = common.ClusterManager.cluster_list[task_cluster_prediction].PE_list[task_PE_prediction]

                    ## Update PE available time
                    PE_comm_wait_times = []

                    for predecessor in shortest_task.preds :
                        # data required from the predecessor for $ready_task
                        for ii, job in enumerate(self.jobs.list):
                            if job.name == shortest_task.jobname:
                                job_ID = ii
                        c_vol = self.jobs.list[job_ID].comm_vol[predecessor, shortest_task.base_ID]
                        
                        # retrieve the real ID  of the predecessor based on the job ID
                        real_predecessor_ID = predecessor + shortest_task.ID - shortest_task.base_ID
                        
                        # Initialize following two variables which will be used if 
                        # PE to PE communication is utilized
                        predecessor_PE_ID = -1
                        predecessor_finish_time = -1
                        
                        for completed in common.TaskQueues.completed.list:
                            if completed.ID == real_predecessor_ID:
                                predecessor_PE_ID = completed.PE_ID
                                predecessor_finish_time = completed.finish_time
                                #print(predecessor, predecessor_finish_time, predecessor_PE_ID)
                        
                        if (common.PE_to_PE):
                            # Compute the PE to PE communication time
                            PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, shortest_task.PE_ID]
                            PE_to_PE_comm_time = int(c_vol/PE_to_PE_band)

                        PE_comm_wait_times.append(max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))
                    ## for predecessor in task.preds :
                                    
                    predecessor_ready_time = max(PE_comm_wait_times, default=0)
                    PE_wait_time = max((self.PEs[shortest_task.PE_ID].available_time - self.env.now), 0)

                    reconfig_latency = DASH_acc_utils.get_DAP_reconfiguration_overhead(self.PEs[i], self.resource_matrix.list[self.PEs[i].ID], task)

                    # update the comparison vector accordingly    
                    ind = self.resource_matrix.list[shortest_task.PE_ID].supported_functionalities.index(shortest_task.name)
                    expected_latency  = self.resource_matrix.list[shortest_task.PE_ID].performance[ind] + max(max(predecessor_ready_time, PE_wait_time), reconfig_latency)

                    ## Enable data aggregation if flag is set to active
                    if common.ils_enable_dagger == True :

                        ##################################################################################################
                        ## Online-ETF to construct on-the-fly Oracle
                        ##################################################################################################
                        comparison = [np.inf]*len(self.PEs)                                     # Initialize the comparison vector 
                        comm_ready = [0]*len(self.PEs)                                          # A list to store the max communication times for each PE
                        
                        for i in range(len(self.resource_matrix.list)):
                            # if the task is supported by the resource, retrieve the index of the task
                            if (shortest_task.name in self.resource_matrix.list[i].supported_functionalities):
                                ind = self.resource_matrix.list[i].supported_functionalities.index(shortest_task.name)
                                
                                    
                                # $PE_comm_wait_times is a list to store the estimated communication time 
                                # (or the remaining communication time) of all predecessors of a task for a PE
                                # As simulation forwards, relevant data is being sent after a task is completed
                                # based on the time instance, one should consider either whole communication
                                # time or the remaining communication time for scheduling
                                PE_comm_wait_times = []
                                
                                # $PE_wait_time is a list to store the estimated wait times for a PE
                                # till that PE is available if the PE is currently running a task
                                PE_wait_time = []
                                  
                                job_ID = -1                                                     # Initialize the job ID
                                
                                # Retrieve the job ID which the current task belongs to
                                for ii, job in enumerate(self.jobs.list):
                                    if job.name == shortest_task.jobname:
                                        job_ID = ii
                                        
                                for predecessor in self.jobs.list[job_ID].task_list[shortest_task.base_ID].predecessors:
                                    # data required from the predecessor for $ready_task
                                    c_vol = self.jobs.list[job_ID].comm_vol[predecessor, shortest_task.base_ID]
                                    
                                    # retrieve the real ID  of the predecessor based on the job ID
                                    real_predecessor_ID = predecessor + shortest_task.ID - shortest_task.base_ID
                                    
                                    # Initialize following two variables which will be used if 
                                    # PE to PE communication is utilized
                                    predecessor_PE_ID = -1
                                    predecessor_finish_time = -1
                                    
                                    
                                    for completed in common.TaskQueues.completed.list:
                                        if (completed.ID == real_predecessor_ID):
                                            predecessor_PE_ID = completed.PE_ID
                                            predecessor_finish_time = completed.finish_time
                                            #print(predecessor, predecessor_finish_time, predecessor_PE_ID)
                                            
                                    
                                    if (common.PE_to_PE):
                                        # Compute the PE to PE communication time
                                        #PE_to_PE_band = self.resource_matrix.comm_band[predecessor_PE_ID, i]
                                        PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, i]
                                        PE_to_PE_comm_time = int(c_vol/PE_to_PE_band)
                                        
                                        PE_comm_wait_times.append(max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))
                                        
                                        if (common.DEBUG_SCH):
                                            print('[D] Time %s: Estimated communication time between PE-%s to PE-%s from task %s to task %s is %d' 
                                                  %(self.env.now, predecessor_PE_ID, i, real_predecessor_ID, shortest_task.ID, PE_comm_wait_times[-1]))
                                        
                                    if (common.shared_memory):
                                        # Compute the communication time considering the shared memory
                                        # only consider memory to PE communication time
                                        # since the task passed the 1st phase (PE to memory communication)
                                        # and its status changed to ready 
                                        
                                        #PE_to_memory_band = self.resource_matrix.comm_band[predecessor_PE_ID, -1]
                                        memory_to_PE_band = common.ResourceManager.comm_band[self.resource_matrix.list[-1].ID, i]
                                        shared_memory_comm_time = int(c_vol/memory_to_PE_band)
                                        
                                        PE_comm_wait_times.append(shared_memory_comm_time)
                                        if (common.DEBUG_SCH):
                                            print('[D] Time %s: Estimated communication time between memory to PE-%s from task %s to task %s is %d' 
                                                  %(self.env.now, i, real_predecessor_ID, shortest_task.ID, PE_comm_wait_times[-1]))
                                    
                                    # $comm_ready contains the estimated communication time 
                                    # for the resource in consideration for scheduling
                                    # maximum value is chosen since it represents the time required for all
                                    # data becomes available for the resource. 
                                    comm_ready[i] = (max(PE_comm_wait_times))
                                    
                                # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors: 
                                
                                # if a resource currently is executing a task, then the estimated remaining time
                                # for the task completion should be considered during scheduling
                                PE_wait_time.append(max((self.PEs[i].available_time - self.env.now), 0))

                                reconfig_latency = DASH_acc_utils.get_DAP_reconfiguration_overhead(self.PEs[i], self.resource_matrix.list[self.PEs[i].ID], task)
                                
                                # update the comparison vector accordingly    
                                comparison[i] = self.resource_matrix.list[i].performance[ind] + max(max(comm_ready[i], PE_wait_time[-1]), reconfig_latency)
    
                                # after going over each resource, choose the one which gives the minimum result
                                resource_id = comparison.index(min(comparison))
                            # end of if (task.name in self.resource_matrix.list[i]...
                        ## for i in range(len(self.resource_matrix.list)):

                        resource_label = resource_id
                        cluster_label  = ils_scripts.get_cluster(resource_label)

                        ## Initialize flag to decide if PE sample should be aggregated
                        aggregate_PE_sample = 0

                        ## Check if cluster predict matches Oracle
                        if cluster_label != task_cluster_prediction :

                            ## Write data onto file 
                            ils_clustera_fp.write(str(self.env.now) + ',')
                            ils_clustera_fp.write(str(shortest_task.ID))
                            for feature in cluster_free_times :
                                ils_clustera_fp.write(',' + str(feature))
                            for feature in il_features :
                                ils_clustera_fp.write(',' + str(feature))
                            ils_clustera_fp.write(',' + str(resource_label))
                            ils_clustera_fp.write(',' + str(cluster_label))
                            ils_clustera_fp.write('\n')

                            ## Check for PE prediction with correct cluster
                            ## Previously we were searching in the incorrect cluster space
                            ## Pass features to the model and predict PE within cluster

                            ## Process features to feed to model
                            features_to_task_model = cluster_PE_free_times[cluster_label]
                            features_to_task_model = np.append(features_to_task_model, np.array(il_features)).reshape(1,-1)

                            ## Load PE model
                            if common.ils_classifier_type == 'RT' :
                                il_PE_model = pickle.load(open('./models/RT_cluster' + str(cluster_label) + '_merged' + common.model_dagger_string + '_model_' + str(common.ils_RT_tree_depth) + '.sav', 'rb'))
                            else :
                                il_PE_model = load_model('./models/NN_cluster' + str(cluster_label) + '_merged' + common.model_dagger_string + '_model' + '.h5')
                            ## if common.ils_classifier_type == 'RT' :

                            ## Use policy to make scheduling decisions for PE within clsuter
                            if common.ils_classifier_type == 'NN' :
                                task_PE_prediction = np.argmax(il_PE_model.predict(features_to_PE_model))
                            else :
                                task_PE_prediction = int(np.around(il_PE_model.predict(features_to_PE_model)[0]))
                            ## if common.ils_classifier_type == 'NN' :

                            task_PE_prediction = common.ClusterManager.cluster_list[cluster_label].PE_list[task_PE_prediction]

                            ## Set flag to aggregate PE sample if PE prediction is incorrect
                            if task_PE_prediction != resource_label :
                                aggregate_PE_sample = 1

                        else :

                            if task_PE_prediction != resource_label :
                                aggregate_PE_sample = 1

                        ## if cluster_label != task_cluster_prediction :

                        if aggregate_PE_sample == 1 :

                            file_handle = vars()['ils_cluster' + str(cluster_label) + '_fp']
                            file_handle.write(str(self.env.now) + ',')
                            file_handle.write(str(shortest_task.ID))
                            for feature in cluster_PE_free_times[cluster_label] :
                                file_handle.write(',' + str(feature))
                            for feature in il_features :
                                file_handle.write(',' + str(feature))
                            file_handle.write(',' + str(resource_label))
                            file_handle.write(',' + str(cluster_label))
                            file_handle.write('\n')
                        ## if aggregate_PE_sample == 1 :

                        ##################################################################################################
                        ## End of Online-ETF to construct on-the-fly Oracle
                        ##################################################################################################
                    
                    ## if common.ils_enable_dagger == True :
                ## if common.ils_enable_policy_decision :

            ## if common.ils_enable_dataset_save or common.ils_enable_policy_decision :

            ##################################################################################################
            ## End of IL-Scheduler
            ##################################################################################################

            # Finally, update the estimated available time of the resource to which
            # a task is just assigned
            # self.PEs[shortest_task.PE_ID].available_time = self.env.now + shortest_comparison[shortest_task.PE_ID]
            self.PEs[shortest_task.PE_ID].available_time = self.env.now + expected_latency + common.scheduling_overhead
            shortest_task.scheduling_overhead = common.scheduling_overhead
            # Remove the task which got a schedule successfully
            for i, task in enumerate(ready_list) :
                if task.ID == shortest_task.ID :
                    ready_list.remove(task)
            
            task_index += 1
            # At the end of this loop, we should have a valid (non-negative ID)
            # that can run next_task

        # end of while len(ready_list) > 0 :

        ## Close file handles
        if common.ils_enable_dataset_save :
            ils_clustera_fp.close()
            for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :
                ## Skip processing for memory
                if cluster.name == 'MEMORY' :
                    continue
                ## if cluster.name == 'MEMORY' :

                vars()['ils_cluster' + str(cluster_index) + '_fp'].close()
            ## for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :
        ## if common.ils_enable_dataset_save :

    #end of ETF( list_of_ready)
    
    def LUT(self,list_of_ready):
        '''
        This is Lookup Table (LUT) scheduler.
        To use this as a standalone version, you need to give 
        predefined task to PE assignments. 
        It is also used by DAS framework.
        '''
        # print(self.env.now)
        if (common.das_dataset and self.env.now >= common.warmup_period):
            ## Set filenames
            ils_clustera_filename = './datasets/' + common.das_dataset_folder + '/data_DAS_' 
            for i in range(len(common.scale_values_list)):
                ils_clustera_filename += str(common.scale_values_list[i])
            for prob in common.job_probabilities:
                ils_clustera_filename += '_' + str(prob)
            ils_clustera_filename += '.csv'
            ils_clustera_fp = open(ils_clustera_filename, 'a')  
            ## Get minimum and maximum job ID of ready tasks
            job_id_list = []
            for task in list_of_ready :
                if task.jobID not in job_id_list :
                    job_id_list.append(task.jobID)
                ## if task.jobID not in job_id_list :
            ## for task in list_of_ready :
            job_id_list = np.array(job_id_list)
            min_job_id = np.min(job_id_list)
            max_job_id = np.max(job_id_list)
            ready_list_for_etf_comparison = copy.deepcopy(list_of_ready)
            self.ETF_DAS(ready_list_for_etf_comparison)
            common.scheduling_overhead = common.set_scheduling_overhead('LUT')  
        for task_index,task in enumerate(list_of_ready):
            task.lookup_mapping = common.table[task.jobname][task.base_ID]                    
            min_cluster = []
            for pe_id_in_cluster in common.ClusterManager.cluster_list[task.lookup_mapping].PE_list:
                min_cluster.append(max(self.PEs[pe_id_in_cluster].available_time - self.env.now, 0))
            if common.das_dataset and self.env.now >= common.warmup_period:
                for i in range(len(list_of_ready)):
                    if ready_list_for_etf_comparison[i].ID == task.ID:
                        etf_decision = ils_scripts.get_cluster(ready_list_for_etf_comparison[i].PE_ID)
                        break 
            task.PE_ID = min_cluster.index(min(min_cluster)) + common.ClusterManager.cluster_list[task.lookup_mapping].PE_list[0] 
            ind = self.resource_matrix.list[task.PE_ID].supported_functionalities.index(task.name)
            PE_comm_wait_times = []
            PE_comm_wait_time = 0
            PE_wait_time = []
            job_ID = -1                                                     # Initialize the job ID
            comm_ready = 0         
            # Retrieve the job ID which the current task belongs to
            for ii, job in enumerate(self.jobs.list):
                if job.name == task.jobname:
                    job_ID = ii
            for predecessor in self.jobs.list[job_ID].task_list[task.base_ID].predecessors:
                # data required from the predecessor for $ready_task
                c_vol = self.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]
                
                # retrieve the real ID  of the predecessor based on the job ID
                real_predecessor_ID = predecessor + task.ID - task.base_ID
                
                # Initialize following two variables which will be used if 
                # PE to PE communication is utilized
                predecessor_PE_ID = -1
                predecessor_finish_time = -1
                
                for completed in common.TaskQueues.completed.list:
                    if (completed.ID == real_predecessor_ID):
                        predecessor_PE_ID = completed.PE_ID
                        predecessor_finish_time = completed.finish_time
                        #print(predecessor, predecessor_finish_time, predecessor_PE_ID)
                
                if (common.PE_to_PE):
                    PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, task.PE_ID]
                    # PE_to_PE_comm_time = int(c_vol/PE_to_PE_band)

                    if common.comm_model == 'PE' :
                        PE_to_PE_comm_time = int(c_vol/PE_to_PE_band)
                    elif common.comm_model == 'NoC' :
                        ## Compute communication time
                        ## If source and destination are same PE, then communication time is assumed to be zero
                        ## If not, compute the time as the number of Manhattan hops on the 2D-mesh
                        pred_cluster = self.resource_matrix.list[predecessor_PE_ID].cluster_ID
                        task_cluster = self.resource_matrix.list[task.PE_ID].cluster_ID

                        if pred_cluster == task_cluster :
                            PE_to_PE_comm_time = 0
                        else :
                            ## Get source and destination mesh positions and x,y indices
                            src_mesh_pos = self.resource_matrix.list[predecessor_PE_ID].position
                            dst_mesh_pos = self.resource_matrix.list[task.PE_ID].position

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

                            PE_to_PE_comm_time = num_hops + (task.input_packet_size - 1)
                    ## if common.comm_model == 'PE' :

                    PE_comm_wait_times.append(max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))                    
                    if (common.DEBUG_SCH):
                        print('[D] Time %s: Estimated communication time between PE-%s to PE-%s from task %s to task %s is %d' 
                              %(self.env.now, predecessor_PE_ID, task.PE_ID, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))
            if PE_comm_wait_times != []:
                PE_comm_wait_time = max(PE_comm_wait_times)                
            comm_ready = max((PE_comm_wait_time),0)
                
            # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors: 
            
            # if a resource currently is executing a task, then the estimated remaining time
            # for the task completion should be considered during scheduling
            PE_wait_time = max((self.PEs[task.PE_ID].available_time - self.env.now), 0)
            
            # update the comparison vector accordingly    
            comparison = self.resource_matrix.list[task.PE_ID].performance[ind] + max(comm_ready, PE_wait_time)      
            if common.das_dataset:
                if len(common.das_inj_rate_sr) != 0 :
                    if common.das_inj_rate_sr[-1] != common.results.total_execution_time:
                        common.das_inj_rate_sr.append(common.results.total_execution_time)  
                        #print(common.results.total_execution_time)
                else:
                    common.das_inj_rate_sr.append(common.results.total_execution_time) 
            if common.das_dataset and self.env.now >= common.warmup_period :
                #print(self.env.now)
                cluster_power_values = []
                for cluster_ind,cluster in enumerate(common.ClusterManager.cluster_list):
                    
                    if cluster.type == 'BIG' or cluster.type == 'LTL':
                        curr_frequency = common.ClusterManager.cluster_list[self.PEs[cluster.PE_list[0]].cluster_ID].current_frequency
                        power = common.ClusterManager.cluster_list[self.PEs[cluster.PE_list[0]].cluster_ID].power_profile[curr_frequency][-1]
                        cluster_power_values.append(power)
                    if cluster.type == 'ACC':
                        dynamic_power = 0
                        for config in common.ClusterManager.cluster_list[self.PEs[cluster.PE_list[0]].cluster_ID].current_acc_kernels_exec:
                            acc_info = DTPM_power_models.get_acc_config_info(common.ClusterManager.cluster_list[self.PEs[cluster.PE_list[0]].cluster_ID].name, config)
                            num_task_in_cls = DASH_Sim_utils.get_num_tasks_being_executed(common.ClusterManager.cluster_list[self.PEs[cluster.PE_list[0]].cluster_ID], self.PEs)
                            if num_task_in_cls != 0:
                                dynamic_power += acc_info.dynamic_power / DASH_Sim_utils.get_num_tasks_being_executed(common.ClusterManager.cluster_list[self.PEs[cluster.PE_list[0]].cluster_ID], self.PEs)
                        cluster_power_values.append(dynamic_power)
                
                ## Collect cluster available times
                cluster_free_times = []
                cluster_PE_free_times = []
                
                ## Track max PEs within each cluster
                max_PEs_in_cluster = 0
                
                ## Iterate through all clusters to populate availability times
                for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :
                
                    ## Skip processing for memory
                    if 'MEM' in cluster.name or 'CAC' in cluster.name :
                        continue
                    ## if 'MEM' in cluster.name or 'CAC' in cluster.name :
                
                    ## Track max PEs within cluster
                    if cluster.num_total_cores > max_PEs_in_cluster :
                        max_PEs_in_cluster = cluster.num_total_cores
                    ## if cluster.num_total_cores > max_PEs_in_cluster :
                
                    ## Get free times of PEs within cluster
                    PE_free_times = []
                    for PE in self.PEs :
                        if PE.ID in cluster.PE_list :
                           ## The available time is divided by 30 for scaling
                           free_time = (max(PE.available_time - self.env.now, 0)) / 30
                           PE_free_times.append(free_time)
                        ## if PE.ID in cluster.PE_list :
                    ## for PE in self.PEs :
                        
                    ## Populate free time of cluster as the earliest time a PE is available within the cluster
                    cluster_free_times.append(min(PE_free_times))
                        
                    PE_free_times = np.pad(PE_free_times, (0, (5 - len(PE_free_times))), constant_values=(10), mode='constant')
                    
                    ## Populate free time of PEs within cluster
                    cluster_PE_free_times.append(PE_free_times)
                    
                ## for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :
                
                ## Convert arrays to numpy arrays
                cluster_free_times    = np.array(cluster_free_times)
                cluster_PE_free_times = np.array(cluster_PE_free_times)

                for ii, job in enumerate(self.jobs.list):
                    if job.name == task.jobname:
                        job_ID = ii

                normalized_task_depth = np.zeros((1, 1))
                normalized_task_ID = np.zeros((1, 1))
                ## Compute node execution times
                #for job_task in task_list :
                task_state = []
                for cluster_ind,cluster in enumerate(common.ClusterManager.cluster_list):                    
                    for PE_index in range(len(self.resource_matrix.list)) :
                        if PE_index == cluster.PE_list[0] and 'MEM' not in cluster.name and 'CAC' not in cluster.name :
                        #if PE_index == 0 or PE_index == 4 or PE_index == 8 or PE_index == 12 or PE_index == 16 or PE_index == 20 or PE_index == 21:
                            if self.jobs.list[job_ID].task_list[task.base_ID].name in self.resource_matrix.list[PE_index].supported_functionalities :
                                index = self.resource_matrix.list[PE_index].supported_functionalities.index(self.jobs.list[job_ID].task_list[task.base_ID].name)
                                task_execution_time = self.resource_matrix.list[PE_index].performance[index]
                            else :
                                task_execution_time = 10000000
                            ## if task.name in self.resource_matrix.list[PE_index].supported_functionalities :
    
                            ## Append execution time to list
                            task_state.append(task_execution_time)
                            break
                        ## if PE_index == 0 or PE_index == 4 or PE_index == 8 or PE_index == 10 or PE_index == 14 :
                    ## for PE_index in range(len(self.resource_matrix.list)) :
                task_state = np.array(task_state)
                if len(task_state[np.all([task_state != 0])]) == 0 :
                    normalized_task_state = np.array([0.0,0.0,10.0,10.0,10.0,10.0,10.0])
                else :
                    normalized_task_state = ils_scripts.get_normalized_list(task_state)
                ## if len(task_state[np.all([task_state != 0, task_state != 10000000])]) == 0 :
                
                ## Append execution times of task to master list
                for job_index, job in enumerate(self.jobs.list) :
                    if task.jobname == job.name :
                        job_depth  = self.jobs.list[job_index].dag_depth['DAG']
                        num_tasks  = len(self.jobs.list[job_index].task_list)
                        break

                
                ## Populate relative job ID
                if max_job_id - min_job_id == 0 :
                    relative_job_id = 0
                else :
                    relative_job_id = (task.jobID - min_job_id) / (max_job_id - min_job_id)
                ## if max_job_id - min_job_id == 0 :            
            
                # Populate job type
                #print(task.jobname)
                if task.jobname == 'ACUMEN' :
                    job_type = 0
                if task.jobname == 'WiFi_Transmitter' :
                    job_type = 1
                if task.jobname == 'WiFi_Receiver' :
                    job_type = 2
                if task.jobname == 'lag_detection' :
                    job_type = 3
                if task.jobname == 'Temporal_Mitigation' :
                    job_type = 4

                ## Populate predecessor PE IDs
                task_preds = []
                task_preds_comm_vol_list = []
                task_preds_cluster_list = []
                task_preds_comm_vol = []
                task_preds_cluster = []
                task_preds.append(self.jobs.list[job_ID].task_list[task.base_ID].preds)
                for task_pred in task_preds[-1]:
                    for completed_task in common.TaskQueues.completed.list :
                        if completed_task.ID == task.head_ID + task_pred :
                            task_preds_cluster.append(ils_scripts.get_cluster(completed_task.PE_ID))
                            c_vol = self.jobs.list[job_ID].comm_vol[task_pred, task.base_ID]
                            task_preds_comm_vol.append(c_vol / 1000)
                        ## for PE in self.PEs :
                    ## if completed_task.ID == task_pred :
                ## for completed_task in common.TaskQueues.completed.list :
            ## for task_pred in task_preds :
                if len(task_preds[-1]) >= 5:
                    num_short_preds = 0
                    task_preds[-1] = task_preds[-1][:5]
                    task_preds_comm_vol = task_preds_comm_vol[:5]
                    task_preds_cluster = task_preds_cluster[:5]
                else:    
                    num_short_preds = 5 - len(task_preds[-1])
                for num in range(num_short_preds) :
                    task_preds[-1] = np.append(task_preds[-1], 10000)
                    task_preds_cluster.append(10000)
                    task_preds_comm_vol.append(10000)
                ## for num in range(num_short_preds) :
#                    task_preds_list.append(task_preds)
                task_preds_cluster_list.append(task_preds_cluster)
                task_preds_comm_vol_list.append(task_preds_comm_vol)
                task_preds_cluster_list = np.stack(task_preds_cluster_list)
                task_preds_cluster_list = task_preds_cluster_list / 4
                task_preds_cluster_list[task_preds_cluster_list > 1000] = 10
                task_preds = np.stack(task_preds)
                task_preds_comm_vol_list = np.stack(task_preds_comm_vol_list)
                task_preds[task_preds > 1000] = 50
                task_preds_comm_vol_list[task_preds_comm_vol_list > 1000] = 10
                normalized_task_depth[0] = task.dag_depth/job_depth
                normalized_task_ID[0] = task.base_ID / num_tasks
                # Injection rate shift register feature 
                inj_rate_sr = np.array(common.das_inj_rate_sr)

                il_features_1 = np.concatenate((task_preds,task_preds_comm_vol_list,normalized_task_ID,normalized_task_state.reshape(1,7),normalized_task_depth,np.array(relative_job_id).reshape(1,1),np.array(job_type).reshape(1,1),task_preds_cluster_list,np.stack(cluster_power_values).reshape(1,-1)), axis=1)

                il_features = il_features_1

                for index in range(il_features.shape[0]) :
                    ## Write data onto file 
                    real_task_ID = task.base_ID
                    ils_clustera_fp.write(str(self.env.now))
                    ils_clustera_fp.write(',' + str(real_task_ID))

                    for feature in il_features[index,:] :
                        ils_clustera_fp.write(',' + str(feature))

                    for feature in cluster_free_times :
                        ils_clustera_fp.write(',' + str(feature))                            

                    ## Iterate over all clusters, get the num of PEs in each cluster and print their availability times
                    for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :
                        if 'MEM' not in cluster.name and 'CAC' not in cluster.name :
                            numPE_in_cluster = len(common.ClusterManager.cluster_list[cluster_index].PE_list)

                            #for ind,feature in  enumerate(vars()['cluster' + str(cluster_index) + '_free_times'])  :
                            for ind,feature in enumerate(cluster_PE_free_times[cluster_index]):
                                if ind < numPE_in_cluster :
                                    ils_clustera_fp.write(',' + str(feature))

                    for ind,feature in enumerate(inj_rate_sr):
                        ils_clustera_fp.write(','+ str(feature))
                    ils_clustera_fp.write(','+str((len(inj_rate_sr)-1)/(inj_rate_sr[-1]-inj_rate_sr[0])*1000000))
                    ils_clustera_fp.write(','+str((inj_rate_sr[-1]-inj_rate_sr[0])/1000))
                    if task.lookup_mapping == etf_decision:
                    ## Print resource label
                        ils_clustera_fp.write(',' + str(0))
                    else:
                        ils_clustera_fp.write(',' + str(-9))
                    
                    ils_clustera_fp.write('\n')
                
            self.PEs[task.PE_ID].available_time = self.env.now + comparison + common.scheduling_overhead   
            task.scheduling_overhead = common.scheduling_overhead

        if (common.das_dataset and self.env.now >= common.warmup_period ):
            ils_clustera_fp.close()
    # end LUT(self,list_of_ready)

    def DAS(self,list_of_ready):
        '''
        This scheduler uses Dynamic Adaptive Scheduler framework.
        It uses two schedulers based on the given decision from 
        Dynamic Adaptive Scheduler (DAS) preselection model.
        '''        
        
        if (common.das_dataset or common.das_policy) and self.env.now == 0:
            from collections import deque
            common.das_inj_rate_sr = deque(maxlen=common.das_inj_rate_sr.maxlen)
            
        if common.das_policy:
            
            job_id_list = []
            for task1 in list_of_ready :
                if task1.jobID not in job_id_list :
                    job_id_list.append(task1.jobID)
                ## if task.jobID not in job_id_list :
            ## for task in list_of_ready :
            job_id_list = np.array(job_id_list)
            min_job_id = np.min(job_id_list)
            max_job_id = np.max(job_id_list) 
            basic_queue = []
            complex_queue = []
            for task in list_of_ready:            
                cluster_power_values = []
                for cluster_ind,cluster in enumerate(common.ClusterManager.cluster_list):
                    
                    if cluster.type == 'BIG' or cluster.type == 'LTL':
                        curr_frequency = common.ClusterManager.cluster_list[self.PEs[cluster.PE_list[0]].cluster_ID].current_frequency
                        power = common.ClusterManager.cluster_list[self.PEs[cluster.PE_list[0]].cluster_ID].power_profile[curr_frequency][-1]
                        cluster_power_values.append(power)
                    if cluster.type == 'ACC':
                        dynamic_power = 0
                        for config in common.ClusterManager.cluster_list[self.PEs[cluster.PE_list[0]].cluster_ID].current_acc_kernels_exec:
                            acc_info = DTPM_power_models.get_acc_config_info(common.ClusterManager.cluster_list[self.PEs[cluster.PE_list[0]].cluster_ID].name, config)
                            num_task_in_cls = DASH_Sim_utils.get_num_tasks_being_executed(common.ClusterManager.cluster_list[self.PEs[cluster.PE_list[0]].cluster_ID], self.PEs)
                            if num_task_in_cls != 0:
                                dynamic_power += acc_info.dynamic_power / DASH_Sim_utils.get_num_tasks_being_executed(common.ClusterManager.cluster_list[self.PEs[cluster.PE_list[0]].cluster_ID], self.PEs)
                        cluster_power_values.append(dynamic_power)
                
                ## Collect cluster available times
                cluster_free_times = []
                cluster_PE_free_times = []
                
                ## Track max PEs within each cluster
                max_PEs_in_cluster = 0
                
                ## Iterate through all clusters to populate availability times
                for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :
                
                    ## Skip processing for memory
                    if 'MEM' in cluster.name or 'CAC' in cluster.name :
                        continue
                    ## if 'MEM' in cluster.name or 'CAC' in cluster.name :
                
                    ## Track max PEs within cluster
                    if cluster.num_total_cores > max_PEs_in_cluster :
                        max_PEs_in_cluster = cluster.num_total_cores
                    ## if cluster.num_total_cores > max_PEs_in_cluster :
                
                    ## Get free times of PEs within cluster
                    PE_free_times = []
                    for PE in self.PEs :
                        if PE.ID in cluster.PE_list :
                           ## The available time is divided by 30 for scaling
                           free_time = (max(PE.available_time - self.env.now, 0)) / 30
                           PE_free_times.append(free_time)
                        ## if PE.ID in cluster.PE_list :
                    ## for PE in self.PEs :
                        
                    ## Populate free time of cluster as the earliest time a PE is available within the cluster
                    cluster_free_times.append(min(PE_free_times))
                        
                    PE_free_times = np.pad(PE_free_times, (0, (5 - len(PE_free_times))), constant_values=(10), mode='constant')
                    
                    ## Populate free time of PEs within cluster
                    cluster_PE_free_times.append(PE_free_times)
                    
                ## for cluster_index, cluster in enumerate(common.ClusterManager.cluster_list) :
                
                ## Convert arrays to numpy arrays
                cluster_free_times    = np.array(cluster_free_times)
                cluster_PE_free_times = np.array(cluster_PE_free_times)

                for ii, job in enumerate(self.jobs.list):
                    if job.name == task.jobname:
                        job_ID = ii

                normalized_task_depth = np.zeros((1, 1))
                normalized_task_ID = np.zeros((1, 1))
                ## Compute node execution times
                #for job_task in task_list :
                task_state = []
                for cluster_ind,cluster in enumerate(common.ClusterManager.cluster_list):                    
                    for PE_index in range(len(self.resource_matrix.list)) :
                        if PE_index == cluster.PE_list[0] and 'MEM' not in cluster.name and 'CAC' not in cluster.name :
                        #if PE_index == 0 or PE_index == 4 or PE_index == 8 or PE_index == 12 or PE_index == 16 or PE_index == 20 or PE_index == 21:
                            if self.jobs.list[job_ID].task_list[task.base_ID].name in self.resource_matrix.list[PE_index].supported_functionalities :
                                index = self.resource_matrix.list[PE_index].supported_functionalities.index(self.jobs.list[job_ID].task_list[task.base_ID].name)
                                task_execution_time = self.resource_matrix.list[PE_index].performance[index]
                            else :
                                task_execution_time = 10000000
                            ## if task.name in self.resource_matrix.list[PE_index].supported_functionalities :
    
                            ## Append execution time to list
                            task_state.append(task_execution_time)
                            break
                        ## if PE_index == 0 or PE_index == 4 or PE_index == 8 or PE_index == 10 or PE_index == 14 :
                    ## for PE_index in range(len(self.resource_matrix.list)) :
                task_state = np.array(task_state)
                if len(task_state[np.all([task_state != 0])]) == 0 :
                    normalized_task_state = np.array([0.0,0.0,10.0,10.0,10.0,10.0,10.0])
                else :
                    normalized_task_state = ils_scripts.get_normalized_list(task_state)
                ## if len(task_state[np.all([task_state != 0, task_state != 10000000])]) == 0 :
                
                ## Append execution times of task to master list
                for job_index, job in enumerate(self.jobs.list) :
                    if task.jobname == job.name :
                        job_depth  = self.jobs.list[job_index].dag_depth['DAG']
                        num_tasks  = len(self.jobs.list[job_index].task_list)
                        break

                
                ## Populate relative job ID
                if max_job_id - min_job_id == 0 :
                    relative_job_id = 0
                else :
                    relative_job_id = (task.jobID - min_job_id) / (max_job_id - min_job_id)
                ## if max_job_id - min_job_id == 0 :            
            
                # Populate job type
                #print(task.jobname)
                if task.jobname == 'ACUMEN' :
                    job_type = 0
                if task.jobname == 'WiFi_Transmitter' :
                    job_type = 1
                if task.jobname == 'WiFi_Receiver' :
                    job_type = 2
                if task.jobname == 'Temporal_Mitigation' :
                    job_type = 3
                if task.jobname == 'lag_detection' :
                    job_type = 4
                ## Populate predecessor PE IDs
                task_preds = []
                task_preds_comm_vol_list = []
                task_preds_cluster_list = []
                task_preds_comm_vol = []
                task_preds_cluster = []
                task_preds.append(self.jobs.list[job_ID].task_list[task.base_ID].preds)
                for task_pred in task_preds[-1]:
                    for completed_task in common.TaskQueues.completed.list :
                        if completed_task.ID == task.head_ID + task_pred :
                            task_preds_cluster.append(ils_scripts.get_cluster(completed_task.PE_ID))
                            c_vol = self.jobs.list[job_ID].comm_vol[task_pred, task.base_ID]
                            task_preds_comm_vol.append(c_vol / 1000)
                        ## for PE in self.PEs :
                    ## if completed_task.ID == task_pred :
                ## for completed_task in common.TaskQueues.completed.list :
            ## for task_pred in task_preds :
                if len(task_preds[-1]) >= 5:
                    num_short_preds = 0
                    task_preds[-1] = task_preds[-1][:5]
                    task_preds_comm_vol = task_preds_comm_vol[:5]
                    task_preds_cluster = task_preds_cluster[:5]
                else:    
                    num_short_preds = 5 - len(task_preds[-1])
                for num in range(num_short_preds) :
                    task_preds[-1] = np.append(task_preds[-1], 10000)
                    task_preds_cluster.append(10000)
                    task_preds_comm_vol.append(10000)
                if len(task_preds[-1]) != len(task_preds_comm_vol):
                    num_short_preds = 5 - len(task_preds_comm_vol) 
                    for num in range(num_short_preds):
                        task_preds_cluster.append(10000)
                        task_preds_comm_vol.append(10000)                                                              
                ## for num in range(num_short_preds) :
#                    task_preds_list.append(task_preds)
                task_preds_cluster_list.append(task_preds_cluster)
                task_preds_comm_vol_list.append(task_preds_comm_vol)
                task_preds_cluster_list = np.stack(task_preds_cluster_list)
                task_preds_cluster_list = task_preds_cluster_list / 4
                task_preds_cluster_list[task_preds_cluster_list > 1000] = 10
                task_preds = np.stack(task_preds)
                task_preds_comm_vol_list = np.stack(task_preds_comm_vol_list)
                task_preds[task_preds > 1000] = 50
                task_preds_comm_vol_list[task_preds_comm_vol_list > 1000] = 10
                normalized_task_depth[0] = task.dag_depth/job_depth
                normalized_task_ID[0] = task.base_ID / num_tasks                            
                # Injection rate shift register feature based on arrival time of the job
                if len(common.das_inj_rate_sr) != 0 :
                    if common.das_inj_rate_sr[-1] != common.results.total_execution_time:
                        common.das_inj_rate_sr.append(common.results.total_execution_time)  
                        #print(common.results.total_execution_time)
                else:
                    common.das_inj_rate_sr.append(common.results.total_execution_time) 
                inj_rate_sr = np.array(common.das_inj_rate_sr) 
                if len(inj_rate_sr) < common.das_inj_rate_sr.maxlen and len(inj_rate_sr) > 1:
                    inj_rate = (len(inj_rate_sr)-1)/(inj_rate_sr[-1]-inj_rate_sr[0])*1000000
                elif len(inj_rate_sr) == 1:
                    inj_rate = np.array([0])
                else:
                    inj_rate = (common.das_inj_rate_sr.maxlen-1)/(inj_rate_sr[-1]-inj_rate_sr[0])*1000000
                #print(inj_rate)                
                
                il_features_1 = np.concatenate((task_preds,task_preds_comm_vol_list,normalized_task_ID,normalized_task_state.reshape(1,7),normalized_task_depth,np.array(relative_job_id).reshape(1,1),np.array(job_type).reshape(1,1),task_preds_cluster_list,np.stack(cluster_power_values).reshape(1,-1)), axis=1)

                #il_features = il_features_1    
                pe_features = np.concatenate((cluster_free_times,cluster_PE_free_times[0][0:4],cluster_PE_free_times[1][0:4],cluster_PE_free_times[2][0:4],cluster_PE_free_times[3][0:4],cluster_PE_free_times[4][0:4],cluster_PE_free_times[5][0:1],cluster_PE_free_times[6][0:4]),axis=0)
                ## Load cluster model
                il_features = np.concatenate((il_features_1,pe_features.reshape(1,len(pe_features)),inj_rate.reshape(1,1)),axis=1)
                # two_features = np.concatenate((cluster_free_times[1].reshape(1,1),inj_rate.reshape(1,1)),axis=1)
                # il_features = two_features
                das_model = pickle.load(open('./models/'+common.das_model, 'rb'))
                ## Use policy to make scheduling decisions for cluster
                scheduler_prediction = int(das_model.predict(il_features.reshape(1,-1))[0])     
                if scheduler_prediction == 0:
                    basic_queue.append(task)
                else:
                    complex_queue.append(task)
            # print('Scheduler Prediction Lookup: ', len(basic_queue))            
            common.scheduling_overhead = common.set_scheduling_overhead('LUT')
            self.LUT(basic_queue)                
            common.das_low_task_counter += len(basic_queue)
            common.das_scheduling_time += len(basic_queue) * common.scheduling_overhead
            # print('Scheduler Prediction ETF: ', len(complex_queue))                
            common.scheduling_overhead = common.set_scheduling_overhead('ETF',len(complex_queue))
            self.ETF_DAS(complex_queue)
            common.das_scheduling_time += len(complex_queue) * common.scheduling_overhead
            common.das_high_task_counter += len(complex_queue)                                           
        else:
            if common.das_dataset:
                #print('# of Jobs: ',common.results.job_counter,' Overload: Low... Scheduler MET')
                common.scheduling_overhead = common.set_scheduling_overhead('LUT')
                self.LUT(list_of_ready)
                #self.ETF_Short(list_of_ready)
                common.das_low_task_counter += len(list_of_ready) 
                common.das_scheduling_time += len(list_of_ready) * common.scheduling_overhead
            else:                
         ############ ETF ##############################       
                if common.das_complex_only: 
                    #print('# of Jobs: ',common.results.job_counter,' Overload: High... Scheduler ETF')
                    common.scheduling_overhead = common.set_scheduling_overhead('ETF',len(list_of_ready))                                                 
                    self.ETF_DAS(list_of_ready)
                    common.das_high_task_counter += len(list_of_ready)
                    common.das_scheduling_time += len(list_of_ready) * common.scheduling_overhead
         ############ Lookup ##############################       
                else:  
                    #print('# of Jobs: ',common.results.job_counter,' Overload: Low... Scheduler MET')
                    common.scheduling_overhead = common.set_scheduling_overhead('LUT')
                    self.LUT(list_of_ready)
                    #self.ETF_Short(list_of_ready)
                    common.das_low_task_counter += len(list_of_ready)   
                    common.das_scheduling_time += len(list_of_ready) * common.scheduling_overhead
    # end DAS(self,list_of_ready)

    def ETF_DAS(self, list_of_ready):
        '''
        This scheduler is Earliest Task First scheduler 
        with added capabilities like returning to pe states 
        to use it with DAS. Created to seperate it from original design. 
        Use ETF() if you want to use ETF scheduler.
        '''
        common.scheduling_overhead = common.set_scheduling_overhead('ETF',len(list_of_ready))        
        ready_list = copy.deepcopy(list_of_ready)
        task_counter = 0
        if (common.das_dataset and self.env.now >= common.warmup_period):
            ## Get minimum and maximum job ID of ready tasks
            job_id_list = []
            for task in list_of_ready :
                if task.jobID not in job_id_list :
                    job_id_list.append(task.jobID)
                ## if task.jobID not in job_id_list :
            ## for task in list_of_ready :
            job_id_list = np.array(job_id_list)
        # Save PE states before entering the functional part to save and load afterwards
        if (common.das_dataset and self.env.now >= common.warmup_period):                                                                                                                                        
            pe_states = []
            for i in range(len(self.resource_matrix.list)):
                pe_states.append(self.PEs[i].available_time)
        # Iterate through the list of ready tasks until all of them are scheduled
        while len(ready_list) > 0 :
    
            shortest_task_exec_time = np.inf
            shortest_task_pe_id     = -1
            shortest_comparison     = [np.inf] * len(self.PEs)
    
            for task in ready_list:
                
                # if task.PE_ID != -1:
                #     continue
                
                comparison = [np.inf]*len(self.PEs)                                     # Initialize the comparison vector 
                comm_ready = [0]*len(self.PEs)                                          # A list to store the max communication times for each PE
                
                if (common.DEBUG_SCH):
                    print ('[D] Time %s: The scheduler function is called with task %s'
                           %(self.env.now, task.ID))
                    
                for i in range(len(self.resource_matrix.list)):
                    # if the task is supported by the resource, retrieve the index of the task
                    if (task.name in self.resource_matrix.list[i].supported_functionalities):
                        ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)
                        
                            
                        # $PE_comm_wait_times is a list to store the estimated communication time 
                        # (or the remaining communication time) of all predecessors of a task for a PE
                        # As simulation forwards, relevant data is being sent after a task is completed
                        # based on the time instance, one should consider either whole communication
                        # time or the remaining communication time for scheduling
                        PE_comm_wait_times = []
                        
                        # $PE_wait_time is a list to store the estimated wait times for a PE
                        # till that PE is available if the PE is currently running a task
                        PE_wait_time = []
                          
                        job_ID = -1                                                     # Initialize the job ID
                        
                        # Retrieve the job ID which the current task belongs to
                        for ii, job in enumerate(self.jobs.list):
                            if job.name == task.jobname:
                                job_ID = ii
                                
                        for predecessor in self.jobs.list[job_ID].task_list[task.base_ID].predecessors:
                            # data required from the predecessor for $ready_task
                            c_vol = self.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]
                            
                            # retrieve the real ID  of the predecessor based on the job ID
                            real_predecessor_ID = predecessor + task.ID - task.base_ID
                            
                            # Initialize following two variables which will be used if 
                            # PE to PE communication is utilized
                            predecessor_PE_ID = -1
                            predecessor_finish_time = -1
                            
                            
                            for completed in common.TaskQueues.completed.list:
                                if (completed.ID == real_predecessor_ID):
                                    predecessor_PE_ID = completed.PE_ID
                                    predecessor_finish_time = completed.finish_time
                                    #print(predecessor, predecessor_finish_time, predecessor_PE_ID)
                                    
                            
                            if (common.PE_to_PE):
                                # Compute the PE to PE communication time
                                #PE_to_PE_band = self.resource_matrix.comm_band[predecessor_PE_ID, i]
                                PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, i]
                                # PE_to_PE_comm_time = int(c_vol/PE_to_PE_band)

                                if common.comm_model == 'PE' :
                                    PE_to_PE_comm_time = int(c_vol/PE_to_PE_band)
                                elif common.comm_model == 'NoC' :
                                    ## Compute communication time
                                    ## If source and destination are same PE, then communication time is assumed to be zero
                                    ## If not, compute the time as the number of Manhattan hops on the 2D-mesh
                                    pred_cluster = self.resource_matrix.list[predecessor_PE_ID].cluster_ID
                                    task_cluster = self.resource_matrix.list[task.PE_ID].cluster_ID

                                    if pred_cluster == task_cluster :
                                        PE_to_PE_comm_time = 0
                                    else :
                                        ## Get source and destination mesh positions and x,y indices
                                        src_mesh_pos = self.resource_matrix.list[predecessor_PE_ID].position
                                        dst_mesh_pos = self.resource_matrix.list[task.PE_ID].position

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

                                        PE_to_PE_comm_time = num_hops + (task.input_packet_size - 1)
                                ## if common.comm_model == 'PE' :

                                
                                PE_comm_wait_times.append(max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))
                                
                                if (common.DEBUG_SCH):
                                    print('[D] Time %s: Estimated communication time between PE-%s to PE-%s from task %s to task %s is %d' 
                                          %(self.env.now, predecessor_PE_ID, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))
                                
                            if (common.shared_memory):
                                # Compute the communication time considering the shared memory
                                # only consider memory to PE communication time
                                # since the task passed the 1st phase (PE to memory communication)
                                # and its status changed to ready 
                                
                                #PE_to_memory_band = self.resource_matrix.comm_band[predecessor_PE_ID, -1]
                                memory_to_PE_band = common.ResourceManager.comm_band[self.resource_matrix.list[-1].ID, i]
                                shared_memory_comm_time = int(c_vol/memory_to_PE_band)
                                
                                PE_comm_wait_times.append(shared_memory_comm_time)
                                if (common.DEBUG_SCH):
                                    print('[D] Time %s: Estimated communication time between memory to PE-%s from task %s to task %s is %d' 
                                          %(self.env.now, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))
                            
                            # $comm_ready contains the estimated communication time 
                            # for the resource in consideration for scheduling
                            # maximum value is chosen since it represents the time required for all
                            # data becomes available for the resource. 
                            comm_ready[i] = (max(PE_comm_wait_times))
                            
                        # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors: 
                        
                        # if a resource currently is executing a task, then the estimated remaining time
                        # for the task completion should be considered during scheduling
                        PE_wait_time.append(max((self.PEs[i].available_time - self.env.now), 0))
                        
                        # update the comparison vector accordingly    
                        comparison[i] = self.resource_matrix.list[i].performance[ind] + max(comm_ready[i], PE_wait_time[-1])
                        
    
                        # after going over each resource, choose the one which gives the minimum result
                        resource_id = comparison.index(min(comparison))
                        #print('aa',comparison)
                    # end of if (task.name in self.resource_matrix.list[i]...
                    
                # obtain the task ID, resource for the task with earliest finish time 
                # based on the computation 
                #print('bb',comparison)
                if min(comparison) < shortest_task_exec_time :
                    shortest_task_exec_time = min(comparison)
                    shortest_task_pe_id     = resource_id
                    shortest_task           = task
                    shortest_comparison     = comparison
                                
                # end of for i in range(len(self.resource_matrix.list)):
            # end of for task in ready_list:
            
            # assign PE ID of the shortest task 
            index = [i for i,x in enumerate(list_of_ready) if x.ID == shortest_task.ID][0]
            list_of_ready[index].PE_ID = shortest_task_pe_id
            list_of_ready[index].scheduling_overhead = common.scheduling_overhead
            list_of_ready[index], list_of_ready[task_counter] = list_of_ready[task_counter], list_of_ready[index]
            shortest_task.PE_ID        = shortest_task_pe_id
            # common.TaskQueues.scheduled.list.append(shortest_task)
    
            # Retrieve the job ID which the current task belongs to
            for ii, job in enumerate(self.jobs.list):
                if job.name == shortest_task.jobname:
                    job_ID = ii

            if shortest_task.PE_ID == -1:
                print ('[E] Time %s: %s can not be assigned to any resource, please check DASH.SoC.**.txt file'
                       % (self.env.now, shortest_task.ID))
                print ('[E] or job_**.txt file')
                assert(task.PE_ID >= 0)           
            else: 
                if (common.DEBUG_SCH):
                    print('[D] Time %s: Estimated execution times for each PE with task %s, respectively' 
                              %(self.env.now, shortest_task.ID))
                    print('%12s'%(''), comparison)
                    print ('[D] Time %s: The scheduler assigns task %s to PE-%s: %s'
                           %(self.env.now, shortest_task.ID, shortest_task.PE_ID, 
                             self.resource_matrix.list[shortest_task.PE_ID].name))
          

            # Finally, update the estimated available time of the resource to which
            # a task is just assigned
            self.PEs[shortest_task.PE_ID].available_time = self.env.now + shortest_comparison[shortest_task.PE_ID] + common.scheduling_overhead  
            # Remove the task which got a schedule successfully
            for i, task in enumerate(ready_list) :
                if task.ID == shortest_task.ID :
                    ready_list.remove(task)
            task_counter += 1

            # At the end of this loop, we should have a valid (non-negative ID)
            # that can run next_task

        # end of while len(ready_list) > 0 :
            # Added to only give schedule and not assign it to tasks 
            # and return to PE states before calling this function
        if ((common.das_dataset)and self.env.now >= common.warmup_period):
            for i in range(len(self.resource_matrix.list)):
                self.PEs[i].available_time = pe_states[i]
        # end of ETF_DAS
    
    def CP(self, list_of_ready):
        '''!
        This scheduler utilizes a look-up table for scheduling tasks to a particular processor
        @param list_of_ready: The list of ready tasks
        '''
        for task in list_of_ready:    
            ind = 0
            base =  0
            for item in common.ilp_job_list:
                if item[0] == task.jobID:
                    ind = common.ilp_job_list.index(item)
                    break
            
            previous_job_list = list(range(ind))
            for job in previous_job_list:
                selection = common.ilp_job_list[job][1]
                num_of_tasks = len(self.jobs.list[selection].task_list)
                base += num_of_tasks
            
            #print(task.jobID, base, task.base_ID)
            
            for i, schedule in enumerate(common.table):
            
                if len(common.table) > base:
                    if (task.base_ID + base) == i:
                        task.PE_ID = schedule[0]
                        task.order = schedule[1]
                else:
                    if ( task.ID%num_of_tasks == i):
                        task.PE_ID = schedule[0]
                        task.order = schedule[1]        
         
            
        list_of_ready.sort(key=lambda x: x.order, reverse=False) 
    # def CP_(self, list_of_ready): 
    
