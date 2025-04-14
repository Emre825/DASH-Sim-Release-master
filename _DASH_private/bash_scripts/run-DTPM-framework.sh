#!/bin/bash

##########################################################################################################################
## DTPM framework script
########################################################################################################################## 
## This script runs the entire DTPM framework
## Author: asartor@cmu.edu
##########################################################################################################################

##########################################################################################################################
## Step-1: Change configurations to the IL-DTPM framework
##########################################################################################################################
echo -e "[INFO] Changing configuration to IL-DTPM framework\n"
sed -i "s/add_new_resource resource_type LTL.*/add_new_resource resource_type LTL resource_name A7 resource_ID 0 capacity 4 num_supported_functionalities 94 DVFS_mode imitation-learning/g" config_SoC/SoC.BAL_only.txt
sed -i "s/add_new_resource resource_type BIG.*/add_new_resource resource_type BIG resource_name A15 resource_ID 1 capacity 4 num_supported_functionalities 94 DVFS_mode imitation-learning/g" config_SoC/SoC.BAL_only.txt
cp config_Files/config_file_DTPM.ini ./config_file.ini

##########################################################################################################################
## Step-2: Generate the DTPM dataset - 640K simulations will be executed
##########################################################################################################################
echo -e "[INFO] Generating DTPM dataset\n"
python generate_traces.py

##########################################################################################################################
## Step-3: Run IL DTPM framework
##########################################################################################################################
echo -e "[INFO] Running the IL-DTPM framework\n"
python DTPM_run_dagger.py
mkdir -p reports
cp trace_temperature.csv ./reports/trace_temperature-IL-DTPM.csv
cp trace_IL_predictions.csv ./reports/trace_IL_predictions-IL-DTPM.csv
cp results.csv ./reports/results-IL-DTPM.csv # Each line represents a DAgger iteration
rm results.csv

##########################################################################################################################
## Step-4: Change DVFS mode to performance, for comparison
##########################################################################################################################
echo -e "[INFO] Changing DVFS mode to performance\n"
sed -i "s/add_new_resource resource_type LTL.*/add_new_resource resource_type LTL resource_name A7 resource_ID 0 capacity 4 num_supported_functionalities 94 DVFS_mode performance/g" config_SoC/SoC.BAL_only.txt
sed -i "s/add_new_resource resource_type BIG.*/add_new_resource resource_type BIG resource_name A15 resource_ID 1 capacity 4 num_supported_functionalities 94 DVFS_mode performance/g" config_SoC/SoC.BAL_only.txt

##########################################################################################################################
## Step-5: Run simulation with performance mode
##########################################################################################################################
echo -e "[INFO] Running simulation with performance DVFS mode\n"
python DASH_Sim_v0.py
cp trace_temperature.csv ./reports/trace_temperature-Perf.csv
cp results.csv ./reports/results-Perf.csv
cp config_Files/config_file.ini ./config_file.ini

## Output files
## - Results and additional traces in ./reports/