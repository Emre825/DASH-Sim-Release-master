import re, shutil, tempfile
import os
import sys
sys.path.append('./plots')

##########################################################################################################################
## DTPM framework script
##########################################################################################################################
## This script runs the DTPM framework for a pretrained policy
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
    if not os.path.exists("DTPM_freq.pkl"):
        print("[E] Pretained policies were not found.")
        print("[E] Please get all files from https://1drv.ms/u/s!AtHpLfUwjh-uj_BKAe5kPxQphOhfHg?e=3RMy05 and unzip it inside DASH-Sim root folder.")
        print("[E] File password: DASH_SIM")
        sys.exit()

    ##########################################################################################################################
    ## Change configurations to the DTPM framework
    ##########################################################################################################################
    print("\n[INFO] Changing configuration to DTPM framework")
    shutil.copyfile("config_Files/config_file_DTPM.ini", "./config_file.ini")
    sed_inplace("config_file.ini", "job_list = .*", "job_list = [[2, 4, 1, 1, 2], [2, 4, 3, 0, 1]]")
    import DASH_Sim_v0
    if not os.path.exists('reports'):
        os.makedirs('reports')

    ##########################################################################################################################
    ## Run simulation with performance DVFS mode
    ##########################################################################################################################
    print("\n[INFO] Running simulation with performance DVFS mode")
    change_DVFS_mode("performance")
    DASH_Sim_v0.run_simulator()
    shutil.copyfile("trace_temperature.csv", "./reports/trace_temperature-Perf.csv")
    shutil.copyfile("results.csv", "./reports/results-Perf.csv")
    os.remove("results.csv")

    ##########################################################################################################################
    ## Run simulation with ondemand DVFS mode
    ##########################################################################################################################
    print("\n[INFO] Running simulation with ondemand DVFS mode")
    change_DVFS_mode("ondemand")
    DASH_Sim_v0.run_simulator()
    shutil.copyfile("trace_temperature.csv", "./reports/trace_temperature-OD.csv")
    shutil.copyfile("results.csv", "./reports/results-OD.csv")
    os.remove("results.csv")

    ##########################################################################################################################
    ## Run simulation with powersave DVFS mode
    ##########################################################################################################################
    print("\n[INFO] Running simulation with powersave DVFS mode")
    change_DVFS_mode("powersave")
    DASH_Sim_v0.run_simulator()
    shutil.copyfile("trace_temperature.csv", "./reports/trace_temperature-Powersave.csv")
    shutil.copyfile("results.csv", "./reports/results-Powersave.csv")
    os.remove("results.csv")

    ##########################################################################################################################
    ## Run IL DTPM framework with a pretrained policy
    ##########################################################################################################################
    print("\n[INFO] Running the IL-DTPM framework")
    change_DVFS_mode("imitation-learning")
    DASH_Sim_v0.run_simulator()
    shutil.copyfile("trace_temperature.csv", "./reports/trace_temperature-IL-DTPM.csv")
    shutil.copyfile("trace_IL_predictions.csv", "./reports/trace_IL_predictions-IL-DTPM.csv")
    shutil.copyfile("results.csv", "./reports/results-IL-DTPM.csv") # Each line in the CSV file represents a DAgger iteration
    os.remove("results.csv")

    ##########################################################################################################################
    ## Restore default configurations
    ##########################################################################################################################
    change_DVFS_mode("performance")
    shutil.copyfile("config_Files/config_file.ini", "./config_file.ini")

    ##########################################################################################################################
    ## Plot the results
    ##########################################################################################################################
    print("\n[INFO] Plotting the results")
    import plot_DTPM_example
    plot_DTPM_example.plot_perf_ener_edp(run_script=True)

    ## Output files
    ## - Results and additional traces in ./reports/