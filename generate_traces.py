'''!
@brief This file contains the script to generate the traces of several configurations at once.

This file is mainly used to create the dataset for the DTPM policies (first step).
The configurations for the trace generation can be configured in this file, including the frequencies/num_cores to be evaluated.
These configurations must match the SoC file defined in the config_file.ini.
This script generates all possible combinations for the provided SoC configurations, for instance, each frequency point of the big cluster will be evaluated with all frequency points of the LITTLE cluster.
'''

import os
import sys
import fnmatch
import pandas as pd
import csv
import time
from shutil import copyfile

import common
import DASH_Sim_utils
import configparser
import DASH_SoC_parser
import DTPM_utils

# Test ondemand, performance, and powersave results
test_individual_configs = False
# Define the DVFS modes to generate the traces
heterogeneous_PEs = True
SoC_file = "MULTIPLE_BAL_SMALL" # MULTIPLE_BAL_SMALL, MULTIPLE_BAL

N_jobs = 10 # Snippet_size
N_applications = 5 # Must match the applications provided in job_file of config_file.ini

if not heterogeneous_PEs:
    # Homogenerous mode
    DVFS_modes = ['constant-2000',
                  # 'constant-1900',
                  'constant-1800',
                  # 'constant-1700',
                  'constant-1600',
                  # 'constant-1500',
                  'constant-1400',
                  # 'constant-1300',
                  'constant-1200',
                  # 'constant-1100',
                  'constant-1000',
                  'constant-800',
                  'constant-600'  # ,
                  # 'constant-400',
                  # 'constant-200'
                  ]
else:
    # Heterogeneous mode
    if SoC_file == "MULTIPLE_BAL_SMALL":
        DVFS_modes = [
            ['constant-1400', 'constant-1200', 'constant-1000', 'constant-800', 'constant-600'],  # 0 (LTL)
            ['constant-2000', 'constant-1800', 'constant-1600', 'constant-1400', 'constant-1200', 'constant-1000', 'constant-800', 'constant-600']  # 1 (BIG)
        ]
    elif SoC_file == "MULTIPLE_BAL":
        DVFS_modes = [
            ['constant-1400', 'constant-1200', 'constant-1000', 'constant-800', 'constant-600'],  # 0 (LTL)
            ['constant-2000', 'constant-1800', 'constant-1600', 'constant-1400', 'constant-1200', 'constant-1000', 'constant-800', 'constant-600'],  # 1 (BIG)
            ['performance'],  # 2 (Scrambler)
            ['performance'],  # 3 (Scrambler)
            ['performance'],  # 4 (FFT)
            ['performance'],  # 5 (FFT)
            ['performance'],  # 6 (FFT)
            ['performance'],  # 7 (FFT)
            ['performance'],  # 8 (Viterbi)
            ['performance']   # 9 (Viterbi)
        ]
    else:
        print("[E] SoC config not found")
        sys.exit()

N_little_list   = [1, 2, 3, 4]
N_big_list      = [1, 2, 3, 4]

resource_matrix = common.ResourceManager()  # This line generates an empty resource matrix

def main():
    start_time = time.time()
    DASH_Sim_utils.clean_traces()
    # Parse the resource file
    config = configparser.ConfigParser()
    config.read('config_file.ini')
    resource_file = "config_SoC/" + config['DEFAULT']['resource_file']
    # Update the number os PEs in the common.py file
    DASH_SoC_parser.resource_parse(resource_matrix, resource_file)  # Parse the input configuration file to populate the resource matrix

    # Run all possible combinations among the frequency points that were generated
    DTPM_utils.run_parallel_sims(DVFS_modes, N_little_list, N_big_list, N_jobs, N_applications, heterogeneous_PEs)

    # Create configuration lists (all PEs have the same DVFS mode) and run the simulator
    if test_individual_configs:
        cfg = ['ondemand', 'performance', 'powersave']
        job_config = [[2, 2, 2, 2, 2]]
        sim_ID = -1
        for c in cfg:
            DVFS_cfg_list = []
            for i in range(common.num_PEs_TRACE):
                DVFS_cfg_list.append(c)
            for N_little in N_little_list:
                for N_big in N_big_list:
                    config_list = (sim_ID, DVFS_cfg_list, job_config, N_little, N_big, common.num_PEs_TRACE)
                    sim_ID -= 1
                    DTPM_utils.run_sim_initial_dataset(config_list)

    # Merge and delete temp csv files - System
    for trace_name in DASH_Sim_utils.trace_list:
        print("Trace name:", trace_name)
        data = set()
        if os.path.exists(trace_name):
            os.remove(trace_name)
        file_list = fnmatch.filter(os.listdir('.'), trace_name.split(".")[0] + '__*.csv')
        if len(file_list) > 0:
            for i, file in enumerate(file_list):
                print("\tFile name:", file)
                if i == 0:
                    with open(file, "r") as f:
                        reader = csv.reader(f)
                        header = next(reader)
                    with open(trace_name, 'w', newline='') as csvfile:
                        wr = csv.writer(csvfile, delimiter=',')
                        wr.writerow(header)
                for i, chunk in enumerate(pd.read_csv(file, chunksize=1000000, iterator=True)):
                    print("\t\tLoading chunk {}...".format(i))
                    data.update(set(chunk.itertuples(index=False, name=None)))
                os.remove(file)
            with open(trace_name, 'a', newline='') as csvfile:
                file_out = csv.writer(csvfile, delimiter=',')
                for line in data:
                    file_out.writerow(list(line))

    copyfile(common.DATASET_FILE_DTPM, common.DATASET_FILE_DTPM.split('.')[0] + " - Initial.csv")

    sim_time = float(float(time.time() - start_time)) / 60.0
    print("--- {:.2f} minutes ---".format(sim_time))

if __name__ == '__main__':
    main()