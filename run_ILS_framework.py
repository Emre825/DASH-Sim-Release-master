##########################################################################################################################
## IL-scheduler framework script
##########################################################################################################################
## This script runs the entire IL-scheduler framework
## Author: anishnk@asu.edu
##########################################################################################################################
import shutil
import re
import tempfile
import runpy
import sys

def sed_inplace(filename, pattern, repl):
    '''
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

##########################################################################################################################
## Step-1: Run DS3 (DASH-Sim) to generate the Oracle and save feature+labels to train IL policies
##########################################################################################################################

## Copy configuration file (corresponding to dataset generation flags)
print()
print("[INFO] Copying configuration file for dataset generation\n")
shutil.copyfile("./config_Files/config_file_ILS_get_dataset.ini", "./config_file.ini")

## Run DS3 (DASH-Sim)
## Approximate runtime: 5.6 minutes on a Xeon Gold 6230 (with 40 physical cores) with Linux OS
## Runtimes on Windows are significantly higher because the "multiprocessing" library is not supported in Windows
print("[INFO] Running DS3 to generate dataset\n")
runpy.run_path("./run_sims_all_injections_ILS.py")

## Output files
## - Execution time reports in ./reports/
## - Dataset generated in ./datasets/

##########################################################################################################################
## Step-2: Merge datasets from multiple injection rates
##########################################################################################################################

## Approximate runtime: < 1 minute
print("[INFO] Merging datasets from all injection rates\n")
runpy.run_path("./ILS_scripts/merge_dataset.py")

## Output files
## - Dataset merged in ./datasets/
## - Relevant files are ./datasets/data_IL_*merged*.csv

##########################################################################################################################
## Step-3: Train IL policies
##########################################################################################################################

## By default, the framework trains Regression Tree (RT) policies with a maximum tree depth of 12
## Approximate runtime: 4.5 minutes on a Xeon Gold 6230 (with 40 physical cores)
## Runtimes on Windows are significantly higher because the "multiprocessing" library is not supported in Windows
print("[INFO] Training IL policies\n")
runpy.run_path("./ILS_scripts/train.py")

##########################################################################################################################
## Step-4: Use the trained IL policies to make scheduling decisions
##########################################################################################################################

## Copy configuration file (corresponding to policy evaluation flags)
print("[INFO] Copying configuration file for policy evaluation\n")
sed_inplace("config_file.ini", "^enable_dataset_save.*", "enable_dataset_save = no")
sed_inplace("config_file.ini", "^enable_ils_policy.*", "enable_ils_policy = yes")

## Run DS3 (DASH-Sim)
## Approximate runtime: 5.0 minutes on a Xeon Gold 6230 (with 40 physical cores)
print("[INFO] Running DS3 for IL policy evaluation")
runpy.run_path("./run_sims_all_injections_ILS.py")

## Output files
## - Execution time reports in ./reports/

##########################################################################################################################
## Step-5: Improve IL policies using data aggregation (DAgger)
##########################################################################################################################

## Copy configuration file (corresponding to policy evaluation flags)
print("[INFO] Copying configuration file for DAgger iterations\n")
sed_inplace("config_file.ini", "^enable_ils_dagger.*", "enable_ils_dagger = yes")

## Run DS3 (DASH-Sim)
## Approximate runtime: 8.0 hours on a Xeon Gold 6230 (with 40 physical cores)
## Runtimes on Windows are significantly higher because the "multiprocessing" library is not supported in Windows
print("[INFO] Running DS3 for IL policy improvement using DAgger")
runpy.run_path("./ILS_scripts/run_dagger.py")

## Output files
## - Execution time reports in ./reports/

## Copy default configuration file
shutil.copyfile("./config_Files/config_file.ini", "./config_file.ini")

##########################################################################################################################
## Step-6: Generate plot
##########################################################################################################################
sys.path.append('./plots/')
print()
print("[INFO] Plotting injection-rate execution-time curve for Oracle and IL-scheduler\n")
print("[INFO] Printing difference in avg. execution time between Oracle and IL-scheduler\n")
import plot_ILS_dagger
