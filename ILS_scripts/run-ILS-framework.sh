#!/bin/bash

##########################################################################################################################
## IL-scheduler framework script
##########################################################################################################################
## This script runs the entire IL-scheduler framework
## Author: anishnk@asu.edu
##########################################################################################################################

##########################################################################################################################
## Step-1: Run DS3 (DASH-Sim) to generate the Oracle and save feature+labels to train IL policies
##########################################################################################################################

## Copy configuration file (corresponding to dataset generation flags)
echo -e ""
echo -e "[INFO] Copying configuration file for dataset generation\n"
cp ./config_Files/config_file_ILS_get_dataset.ini ./config_file.ini

## Run DS3 (DASH-Sim)
## Approximate runtime: 5.6 minutes on a Xeon Gold 6230 (with 40 physical cores)
echo -e "[INFO] Running DS3 to generate dataset\n"
python ./run_sims_all_injections_ILS.py

## Output files
## - Execution time reports in ./reports/
## - Dataset generated in ./datasets/

##########################################################################################################################
## Step-2: Merge datasets from multiple injection rates
##########################################################################################################################

## Approximate runtime: < 1 minute
echo -e "[INFO] Merging datasets from all injection rates\n"
python ./ILS_scripts/merge_dataset.py

## Output files
## - Dataset merged in ./datasets/
## - Relevant files are ./datasets/data_IL_*merged*.csv

##########################################################################################################################
## Step-3: Train IL policies
##########################################################################################################################

## By default, the framework trains Regression Tree (RT) policies with a maximum tree depth of 12
## Approximate runtime: 4.5 minutes on a Xeon Gold 6230 (with 40 physical cores)
echo -e "[INFO] Training IL policies\n"
python ./ILS_scripts/train.py

##########################################################################################################################
## Step-4: Use the trained IL policies to make scheduling decisions
##########################################################################################################################

## Copy configuration file (corresponding to policy evaluation flags)
echo -e "[INFO] Copying configuration file for policy evaluation\n"
cp ./config_Files/config_file_ILS_evaluate.ini ./config_file.ini

## Run DS3 (DASH-Sim)
## Approximate runtime: 5.0 minutes on a Xeon Gold 6230 (with 40 physical cores)
echo -e "[INFO] Running DS3 for IL policy evaluation"
python ./run_sims_all_injections_ILS.py

## Output files
## - Execution time reports in ./reports/

