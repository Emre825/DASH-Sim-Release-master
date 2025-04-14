'''!
@brief This file generates the oracle for the IL-based DTPM policies.

After obtaining a dictionary with the oracle decisions for all configurations, update the dataset files with the corresponding oracle for each sample in the dataset.

This is applied for all three kinds of IL DTPM policies: policy for changing the frequency, number of cores, and the regression policy that predicts the execution time.
'''

import os
import sys
import time
import configparser

import DTPM_utils
import DASH_SoC_parser
import common

resource_matrix = common.ResourceManager()  # This line generates an empty resource matrix
config = configparser.ConfigParser()
config.read('config_file.ini')
resource_file = "config_SoC/" + config['DEFAULT']['resource_file']
DASH_SoC_parser.resource_parse(resource_matrix, resource_file)                  # Parse the input configuration file to populate the resource matrix
start_time = time.time()

# Get oracle dictionary for all configurations in the dataset
common.oracle_config_dict = DTPM_utils.get_oracle_frequencies_and_num_cores()

def generate_oracle():
    '''!
    Based on the obtained oracle dictionary, update all datasets with the oracle decisions for each sample.
    '''
    if os.path.exists(common.DATASET_FILE_DTPM):
        print("Loading dataset and obtaining the oracle frequencies...")
        sim_time = float(float(time.time() - start_time)) / 60.0
        print("{:.2f} minutes".format(sim_time))
        DTPM_utils.add_oracle_to_dataset('Frequency')
        DTPM_utils.create_reduced_dataset('Frequency')
        DTPM_utils.add_oracle_to_dataset('Num_cores')
        DTPM_utils.create_reduced_dataset('Num_cores')
        DTPM_utils.add_oracle_to_dataset('Regression')
        DTPM_utils.create_reduced_dataset('Regression')
        sim_time = float(float(time.time() - start_time)) / 60.0
        print("--- {:.2f} minutes ---".format(sim_time))
        print("DONE: dataset is saved...")
    else:
        print("[E] Dataset file not found:", common.DATASET_FILE_DTPM)
        sys.exit()

if __name__ == '__main__':
    generate_oracle()