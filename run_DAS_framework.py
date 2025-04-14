##########################################################################################################################
## Dynamic Adaptive Scheduler framework script
##########################################################################################################################
## This script runs the entire Dynamic Adaptive Scheduler framework
## Author: agoksoy@wisc.edu
##########################################################################################################################
import shutil
import re
import tempfile
import runpy
import os,sys
import contextlib

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

#########################################################################################################################
#%% Step-1: Run DS3 (DASH-Sim) to generate the dataset and save feature+labels to train DAS policies
##########################################################################################################################

## Copy configuration file (corresponding to dataset generation flags)
print()
print("[INFO] Copying configuration file for dataset generation\n")
shutil.copyfile("./config_Files/config_file_DAS_get_dataset.ini", "./config_file.ini")

## Run DS3 (DASH-Sim)
print("[INFO] Running DS3 to generate dataset\n")
sed_inplace("./run_sims_all_injections_DAS.py", ".*ds3.common.das_dataset.*", "ds3.common.das_dataset = True")
sed_inplace("./run_sims_all_injections_DAS.py", ".*ds3.common.das_complex_only.*", "ds3.common.das_complex_only = False")
sed_inplace("./run_sims_all_injections_DAS.py", ".*ds3.common.das_policy.*", "ds3.common.das_policy = False")
with contextlib.redirect_stdout(None):
    runpy.run_path("./run_sims_all_injections_DAS.py",run_name ='__main__')

# print("[INFO] Copying configuration file for generating ETF results\n")
sed_inplace("./run_sims_all_injections_DAS.py", ".*ds3.common.das_dataset.*", "ds3.common.das_dataset = False")
sed_inplace("./run_sims_all_injections_DAS.py", ".*ds3.common.das_complex_only.*", "ds3.common.das_complex_only = True")
sed_inplace("./run_sims_all_injections_DAS.py", ".*ds3.common.das_policy.*", "ds3.common.das_policy = False")

## Run DS3 (DASH-Sim)
print("[INFO] Running DS3 to generate ETF results for dataset postprocessing\n")
with contextlib.redirect_stdout(None):
    runpy.run_path("./run_sims_all_injections_DAS.py",run_name ='__main__')
    
## Output files
## - Execution time reports in ./reports/
## - Dataset generated in ./datasets/DAS/

##########################################################################################################################
#%% Step-2: Postprocess and Merge datasets from multiple injection rates
##########################################################################################################################

print('[INFO] Postprocessing based on both scheduler results and merging datasets from all injection rates\n')
os.makedirs('./datasets/DAS/postprocessed/', exist_ok=True)
runpy.run_path("./DAS_scripts/dataset_labeling.py")

## Output file
## - Dataset merged in ./datasets/DAS/postprocessed/

##########################################################################################################################
#%% Step-3: Train DAS policies
##########################################################################################################################

## By default, the framework trains Regression Tree (RT) policies
print("[INFO] Training DAS policies\n")
with contextlib.redirect_stdout(None):
    runpy.run_path("./DAS_scripts/train_das_feature_selection.py")
    
##########################################################################################################################
#%% Step-4: Use the trained DAS policies to make scheduling decisions
##########################################################################################################################

## Copy configuration file (corresponding to policy evaluation flags)
print("[INFO] Copying configuration file for policy evaluation\n")
shutil.move('./models/all_8_RT_clustera.sav', './models/das_model.sav')
sed_inplace("./run_sims_all_injections_DAS.py", ".*ds3.common.das_dataset.*", "ds3.common.das_dataset = False")
sed_inplace("./run_sims_all_injections_DAS.py", ".*ds3.common.das_complex_only.*", "ds3.common.das_complex_only = False")
sed_inplace("./run_sims_all_injections_DAS.py", ".*ds3.common.das_policy.*", "ds3.common.das_policy = True")

## Run DS3 (DASH-Sim)
print("[INFO] Running DS3 for DAS policy evaluation")
with contextlib.redirect_stdout(None):
    runpy.run_path("./run_sims_all_injections_DAS.py",run_name ='__main__')

## Output files
## - Execution time reports in ./reports/

## Copy default configuration files and change running scripts back to original
shutil.copyfile("./config_Files/config_file.ini", "./config_file.ini")
shutil.move('./models/das_model.sav', './models/all_8_RT_clustera.sav')
sed_inplace("./run_sims_all_injections_DAS.py", ".*ds3.common.das_dataset.*", "#ds3.common.das_dataset = True")
sed_inplace("./run_sims_all_injections_DAS.py", ".*ds3.common.das_complex_only.*", "#ds3.common.das_complex_only = False")
sed_inplace("./run_sims_all_injections_DAS.py", ".*ds3.common.das_policy.*", "#ds3.common.das_policy = False")

##########################################################################################################################
#%% Step-5: Generate plot
##########################################################################################################################
sys.path.append('./plots/')
print()
print("[INFO] Plotting data-rate execution-time curve for ETF,LUT and DAS\n")
import plot_DAS
