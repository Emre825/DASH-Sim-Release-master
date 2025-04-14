import common
import numpy as np
import configparser
import ctypes
from numpy.ctypeslib import ndpointer


import DASH_SoC_parser                                                          # The resource parameters used in DASH-Sim are obtained from

## Initialize variables
num_rows = 4
num_cols = 4

# Initialize the injection rate matrix
lambda_array = np.zeros((num_rows * num_cols, num_cols * num_rows), dtype=np.float32)

# Initialize the return data type from analytical models
late_matrix = np.zeros((num_rows * num_cols, num_cols * num_rows), dtype=np.float32)
#print(late_matrix)

# Loading the shared object for latency models in C / C++
lib_analytical_models = ctypes.cdll.LoadLibrary('./libwrapperLatencyModels.so')
 
# Defining the types of the arguments of the function defined in C
lib_analytical_models.wrapperLatencyModels.argtypes = [ctypes.c_int, ctypes.c_int, 
         ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), 
         ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

print(lib_analytical_models)


# =============================================================================
# # Generation of the injection rate matrix
# for row_idx in range(num_rows * num_cols) :
#     for col_idx in range(num_cols * num_rows) :
#         lambda_array[row_idx][col_idx] = 0.0
# 
# # Calling the analytical latency model function in C from Python
# lib_analytical_models.wrapperLatencyModels(num_rows, num_cols, lambda_array, late_matrix)
# =============================================================================



Positions = []
Cache_positions = []
MC_position = -1
# Based on positions, create circular buffers (queues) for each source-destination pair
# this buffers with a size of 100 will store packet sizes to be transfed from a source 
# to destination for a 100-cycle window
S2D_buffers = {}
#write_time = -1
#read_time = -1


# Instantiate the ResourceManager object that contains all the resources
# in the target DSSoC
resource_matrix = common.ResourceManager()                                      # This line generates an empty resource matrix
config = configparser.ConfigParser()
config.read('config_file.ini')
resource_file = "config_SoC/" + config['DEFAULT']['resource_file']
DASH_SoC_parser.resource_parse(resource_matrix, resource_file)                  # Parse the input configuration file to populate the resource matrix


# Acquire all positions that have at least one entry (PE, Cache, or Memory Controller)

for resource in (resource_matrix.list):
    if (resource.position != -1) and (int(resource.position) not in Positions):
        Positions.append(int(resource.position))
    if (resource.type == 'CAC') and not(resource.position in Cache_positions):
        Cache_positions.append(int(resource.position)) 
    if (resource.type == 'MEM'):
        MC_position = int(resource.position) 
print(Positions)
print(Cache_positions)
print(MC_position)

for p in Positions:
    for c in Cache_positions:
        common.PE_to_Cache[(p,c)] = 0

# Based on positions, create circular buffers (queues) for each source-destination pair
# this buffers with a size of 100 will store packet sizes to be transfed from a source 
# to destination for a 100-cycle window
S2D_buffers = {(s,d): [0]*100 for s in range(num_rows * num_cols) for d in range(num_rows * num_cols)}
#print(S2D_buffers)


def write_to_cache(env_time,PE_position,cache_position,buffer_ind,completed_task):
    # Accumulate output data of the ready task in $S2D_buffer
    if (common.write_time < common.warmup_period):
        # It is possible that a couple of tasks finish at the same time
        # then, data should might be added into a buffer instead of overwriting
        if (common.write_time == int(env_time)):
            S2D_buffers[(PE_position,cache_position)][buffer_ind] += int(completed_task.output_packet_size)
            if (common.DEBUG_SIM):
                print('[D] Time %d: Output data of the completed task with ID %d is written to cache(%d) from PE-%s(%d)'
                      %(env_time, completed_task.ID, cache_position, completed_task.PE_ID, PE_position))
                print('%10s'%('')+'%s packets will be stored in slot %d of the source(%d) to destination(%d) circular buffer'
                      %(completed_task.output_packet_size,buffer_ind, PE_position, cache_position))
                print('%10s'%('')+'Slot %d value is now'%(buffer_ind),S2D_buffers[(PE_position,cache_position)][buffer_ind])
        else:
            S2D_buffers[(PE_position,cache_position)][buffer_ind] = int(completed_task.output_packet_size)
            if (common.DEBUG_SIM):
                print('[D] Time %d: Output data of the completed task with ID %d is written to cache(%d) from PE-%s(%d)'
                      %(env_time, completed_task.ID, cache_position, completed_task.PE_ID, PE_position))
                print('%10s'%('')+'%s packets will be stored in slot %d of the source(%d) to destination(%d) circular buffer'
                      %(completed_task.output_packet_size,buffer_ind, PE_position, cache_position))
                print('%10s'%('')+'Slot %d value is now'%(buffer_ind),S2D_buffers[(PE_position,cache_position)][buffer_ind])
        common.write_time = int(env_time)
    
def read_from_cache(cache_hit,env_time, cache_position, PE_position, buffer_ind, ready_task):
    # Accumulate input data of the ready task in $S2D_buffer
    if (common.read_time < common.warmup_period):
        # If it is a hit, the packets will come from a cache
        if (cache_hit):
            if (common.read_time == int(env_time)):
                S2D_buffers[(cache_position,PE_position)][buffer_ind] += int(ready_task.input_packet_size)
                if (common.DEBUG_SIM):
                    print('[D] Time %d: Input data of the task with ID %d is written to PE-%s(%d) from cache(%d)'
                          %(env_time, ready_task.ID, ready_task.PE_ID, PE_position, cache_position))
                    print('%10s'%('')+'%s packets will be stored in slot %d of the source(%d) to destination(%d) circular buffer'
                          %(ready_task.input_packet_size,buffer_ind, cache_position, PE_position,))
                    print('%10s'%('')+'Slot %d value is now'%(buffer_ind),S2D_buffers[(cache_position,PE_position)][buffer_ind])
            else:
                S2D_buffers[(cache_position,PE_position)][buffer_ind] = int(ready_task.input_packet_size)
                if (common.DEBUG_SIM):
                    print('[D] Time %d: Input data of the task with ID %d is written to PE-%s(%d) from cache(%d)'
                          %(env_time, ready_task.ID, ready_task.PE_ID, PE_position, cache_position))
                    print('%10s'%('')+'%s packets will be stored in slot %d of the source(%d) to destination(%d) circular buffer'
                          %(ready_task.input_packet_size,buffer_ind, cache_position, PE_position,))
                    print('%10s'%('')+'Slot %d value is now'%(buffer_ind),S2D_buffers[(cache_position,PE_position)][buffer_ind])
        # if it is a miss, the packet should be transferred from DRAM to a cache
        # then the cache will send them to the target PE
        else:
            if (common.read_time == int(env_time)):
                S2D_buffers[(cache_position,PE_position)][buffer_ind] += int(ready_task.input_packet_size)
                S2D_buffers[(MC_position,cache_position)][buffer_ind] += int(ready_task.input_packet_size)
                if (common.DEBUG_SIM):
                    print('[D] Time %d: Input data of the task with ID %d is written to PE-%s(%d) from cache(%d)'
                          %(env_time, ready_task.ID, ready_task.PE_ID, PE_position, cache_position))
                    print('%10s'%('')+'%s packets will be stored in slot %d of the source(%d) to destination(%d) circular buffer'
                          %(ready_task.input_packet_size,buffer_ind, cache_position, PE_position,))
                    print('%10s'%('')+'Slot %d value is now'%(buffer_ind),S2D_buffers[(cache_position,PE_position)][buffer_ind])                        
                    print('[D] Time %d: Input data of the task with ID %d is written to cache(%d) from memory controller(%d)'
                          %(env_time, ready_task.ID, cache_position, MC_position))
                    print('%10s'%('')+'%s packets will be stored in slot %d of the source(%d) to destination(%d) circular buffer'
                          %(ready_task.input_packet_size,buffer_ind, MC_position, cache_position))
                    print('%10s'%('')+'Slot %d value is now'%(buffer_ind),S2D_buffers[(MC_position, cache_position)][buffer_ind])   
            else:
                S2D_buffers[(cache_position,PE_position)][buffer_ind] = int(ready_task.input_packet_size)
                S2D_buffers[(MC_position,cache_position)][buffer_ind] = int(ready_task.input_packet_size)
                if (common.DEBUG_SIM):
                    print('[D] Time %d: Input data of the task with ID %d is written to PE-%s(%d) from cache(%d)'
                          %(env_time, ready_task.ID, ready_task.PE_ID, PE_position, cache_position))
                    print('%10s'%('')+'%s packets will be stored in slot %d of the source(%d) to destination(%d) circular buffer'
                          %(ready_task.input_packet_size,buffer_ind, cache_position, PE_position,))
                    print('%10s'%('')+'Slot %d value is now'%(buffer_ind),S2D_buffers[(cache_position,PE_position)][buffer_ind])                        
                    print('[D] Time %d: Input data of the task with ID %d is written to cache(%d) from memory controller(%d)'
                          %(env_time, ready_task.ID, cache_position, MC_position))
                    print('%10s'%('')+'%s packets will be stored in slot %d of the source(%d) to destination(%d) circular buffer'
                          %(ready_task.input_packet_size,buffer_ind, MC_position, cache_position))
                    print('%10s'%('')+'Slot %d value is now'%(buffer_ind),S2D_buffers[(MC_position, cache_position)][buffer_ind])
        # end of if (cache_hit):
        common.read_time = int(env_time)