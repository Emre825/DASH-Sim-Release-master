import re, shutil, tempfile
import os
import sys

##########################################################################################################################
## DTPM framework script
##########################################################################################################################
## This script runs the entire DTPM framework
## Author: asartor@cmu.edu
##########################################################################################################################

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

def change_DVFS_mode(DVFS_mode):
    sed_inplace("./config_SoC/SoC.BAL_only.txt", "add_new_resource resource_type LTL.*", "add_new_resource resource_type LTL resource_name A7 resource_ID 0 capacity 4 num_supported_functionalities 94 DVFS_mode " + DVFS_mode)
    sed_inplace("./config_SoC/SoC.BAL_only.txt", "add_new_resource resource_type BIG.*", "add_new_resource resource_type BIG resource_name A15 resource_ID 1 capacity 4 num_supported_functionalities 94 DVFS_mode " + DVFS_mode)

if __name__ == '__main__':
    if not os.path.exists("DTPM_hardware_counters_trace_WIFI_TX.csv"):
        print("[E] Hardware counter files were not found.")
        print("[E] Please get all files from https://1drv.ms/u/s!AtHpLfUwjh-uj_BKAe5kPxQphOhfHg?e=3RMy05 and unzip it inside DASH-Sim root folder.")
        print("[E] Zip password: DASH_SIM")
        sys.exit()

    ##########################################################################################################################
    ## Step-1: Change configurations to the IL-DTPM framework
    ##########################################################################################################################
    print("[INFO] Changing configuration to IL-DTPM framework")
    change_DVFS_mode("performance")
    shutil.copyfile("config_Files/config_file_DTPM.ini", "./config_file.ini")

    ##########################################################################################################################
    ## Step-2: Generate the DTPM dataset
    ## 640K simulations will be executed - highly parallel (will use all available CPU cores)
    ## Approximate runtime: 8 hours on a Xeon Gold 6230 (with 40 physical cores)
    ##########################################################################################################################
    print("[INFO] Generating DTPM dataset")
    import generate_traces
    generate_traces.main()

    ##########################################################################################################################
    ## Step-3: Run IL DTPM framework
    ## Approximate runtime: 4 hours on a Xeon Gold 6230 (with 40 physical cores)
    ##########################################################################################################################
    print("[INFO] Running the IL-DTPM framework")
    change_DVFS_mode("imitation-learning")
    import DTPM_run_dagger
    DTPM_run_dagger.main()
    if not os.path.exists('reports'):
        os.makedirs('reports')
    shutil.copyfile("trace_temperature.csv", "./reports/trace_temperature-IL-DTPM.csv")
    shutil.copyfile("trace_IL_predictions.csv", "./reports/trace_IL_predictions-IL-DTPM.csv")
    shutil.copyfile("results.csv", "./reports/results-IL-DTPM.csv") # Each line represents a DAgger iteration
    os.remove("results.csv")

    ##########################################################################################################################
    ## Restore default configurations
    ##########################################################################################################################
    change_DVFS_mode("performance")
    shutil.copyfile("config_Files/config_file.ini", "./config_file.ini")

    ## Output files
    ## - Results and additional traces in ./reports/