*************************************************************************
   	© Copyright 2020 ASU All rights reserved.
      This file contains confidential and proprietary
 	    information of DASH-SoC Project.
*************************************************************************
# 1. Installation Guide :

Please refer to Installation_guide.txt file to install required software.

# 2. How to run the DS3?

DS3 requires the following minimum inputs to run:
1- A SoC configuration file that specifies the target hardware, i.e., the list of processing elements (PEs) in DASH-SoC.
2- The applications(s) that will run on the target hardware, i.e., the list of tasks and their dependencies.
3- The name of the default schedule (e.g., min_execution_time). 

More details about these inputs are provided below.
Please also refer to config_file.ini file for more information about the required formats and examples.

## 2.1 SoC Configuration File (config_SoC folder)
Create a SoC hardware configuration (Resource list) in a text file (SoC.*.txt) using the key word: add_new_resource followed by the resource type, name, ID, capacity, and the list of supported tasks. The capacity specifies the number of concurrent tasks the resource can run.
```
Example 1a: The following lines add a new CPU with name=P1, ID=0, capacity=1, which can run three types of tasks (specified in three consecutive lines): 

add_new_resource resource_type CPU resource_name P1 resource_ID 0 capacity 1 num_supported_functionalities 3
scrambler 12
reed_solomon_encoder 15
bpsk_modulation 18

The numbers next to the task names indicate the execution time of the corresponding taks on this resource in unit time granularity (e.g., microsecond). 
In addition to these inputs, the input specifies the dynamic voltage-freqeuency scaling (DVFS) mode supported by this PE. For example, the following addition sets the DVFS mode as "performance" and specifies the frequency scaling information, explained in the following example.

Example 1b: The following lines add a new CPU with name=P1, ID=0, capacity=1 and that can run 3 different tasks using "performance" DVFS mode
```
add_new_resource resource_type CPU resource_name P1 resource_ID 0 capacity 1 num_supported_functionalities 3 DVFS_mode performance
opp 1000 1150               # operating point frequency in MHz and voltage in mW (1 GHz, 1.15 V in this example)
trip_freq -1 -1 -1          # -1: Thermal throttling not enabled. If thermal throttling is enabled, 
                            # the throttled frequencies are specified
power_profile 1000 0.1      # The power consumption at 1GHz freqeuency. One tuple needed for each supported frequency.
PG_profile 1000 0.1         # The leakage power consumption of the resource when it is gated.
scrambler 12
reed_solomon_encoder 15
bpsk_modulation 18
```

The format general format: 
add_new_resource $resource_type (string)  $resource_name (string) $resource_id (int) $capacity (int) $num_of_supported_functionality (int) $DVFS_mode (string)
    $functionality_name (string) $execution_time (float)
        opp $frequency (int - MHz) $voltage (int - mV), defines the Operating Performance Points (OPPs) with frequency and voltage tuples
        trip_freq $trip_1 $trip_2 $trip_3 ..., defines the frequencies that are set at each trip point if throttling is enabled. "-1" means that the frequency is not modified
        power_profile $frequency $power_1 $power_2 $power_3 ... $power_max_capacity
        PG_profile $frequency $power_1 $power_2 $power_3 ... $power_max_capacity
```

Please see config_SoC/SoC_Top.txt for a detailed example.

## 2.2 Job (Application File, under the config_Jobs folder)
Create a Task list in a text file (job_*.txt) using the key word add_new_tasks followed by the task list.
```
The format: add_new_tasks $num_of_tasks (int)
            $task_name (string) $task_id (int) $task_predecessors (list)
```

Example: The following lines add one new task, whose name, ID, and predecessors are specified in the subsequent line. 
In this example, name: scrambler, ID: 0, and predecessor is another task with ID=2
(empty list means there is no dependency) 
```
add_new_tasks 1
scrambler 0 2
```

Please see config_Jobs/job_Top.txt for a detailed example.

## 2.3 Simulator Configuration File
All remaining SoC configuration parameters are saved in config_file.init. 
It can be modified to adjust the input configurations as needed.

## 2.4 Running the Simulator
Finally, to start the simulation by running:

DASH_Sim_v0.py 

Please be sure that all the files listed below are in your file directory

# 3. File Structure
```
├── DASH_Sim_v0.py               : This file is the main() function which should be run to get the simulation results.
|   ├── ACC_model.csv            : This file contains the information about the accelerators in the SoC.
│   ├── clusters.py              : This file contains the information about the clusters in the SoC.
│   ├── common.py                : This file contains all the common parameters used in DASH_Sim.
│   ├── config_file.ini          : This file contains all the file names and variables to initialize the DASH_Sim
│   ├── CP_models.ini            : This file contains the code for dynamic scheduling with Constraint Programming.
│   ├── DASH_acc_parser.py       : This file contains the code to parse DASH-SoC accelerators given in ACC_model.csv file.
│   ├── DASH_acc_utils.py        : This file contains the functions to manage the execution and configuration of accelerators, including DAP.
│   ├── DASH_Sim_core.py         : This file contains the simulation core that handles the simulation events.
│   ├── DASH_Sim_utils.py        : This file contains functions that are used by DASH_Sim.
│   ├── DASH_SoC_parser.py       : This file contains the code to parse DASH-SoC given in config_file.ini file.
│   ├── DTPM.py                  : This file contains the code for the DTPM module.
│   ├── DTPM_generate_oracle.py  : This file generates the oracle for the IL-based DTPM policies.
│   ├── DTPM_policies.py         : This file contains the DVFS policies.
│   ├── DTPM_power_models.py     : This file contains functions that are used by the DVFS mechanism and PEs to get performance, power, and thermal values.
│   ├── DTPM_train_model.py      : This file contains the training methods for the ML-based DTPM policies.
│   ├── DTPM_utils.py            : This file contains functions that are used by the DTPM module.
│   ├── ils_scripts.py           : This file contains few functions required for IL-scheduler.
│   ├── job_generator.py         : This file contains the code for the job generator.
│   ├── job_parser.py            : This file contains the code to parse jobs given in config_file.ini file.
│   ├── processing_element.py    : This file contains the process elements and their attributes.
│   ├── scheduler.py             : This file contains the code for scheduler class which contains different types of scheduler.
│   ├── config_SoC/SoC.*.txt     : These files are the configuration files of the Resources available in DASH-SoC.
│   └── config_Jobs/job_*.txt    : These files are the configuration files of the Jobs.
└── ...
```
