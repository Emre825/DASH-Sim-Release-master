'''!
@brief This file runs the simulator multiple times and applies DAgger (used by the IL-based DTPM policy)

Step-by-step execution:
- This file prepares the dataset by creating a copy of the initial dataset (if the file ends with " - Initial.csv"), otherwise the current dataset is directly modified.
- Generate the oracle and update the dataset.
- Execute the pre-defined number of DAgger iterations:
    - Train the policy based on the current dataset
    - Call the simulator to execute the given workload and aggregate data to the dataset
'''

import os
from shutil import copyfile
import time

import common
import DASH_Sim_v0
import DASH_Sim_utils
import DTPM_train_model

def main():
    start_time = time.time()

    dataset_oracle = common.DATASET_FILE_DTPM
    dataset_oracle_initial = common.DATASET_FILE_DTPM.split('.')[0] + " - Initial.csv"
    if os.path.exists(dataset_oracle_initial):
        copyfile(dataset_oracle_initial, dataset_oracle)

    import DTPM_generate_oracle
    DTPM_generate_oracle.generate_oracle()

    if os.path.exists(common.RESULTS):
        os.remove(common.RESULTS)

    for iter in range(common.DAgger_iter):
        DASH_Sim_utils.clean_policies()

        common.wrong_predictions_freq = 0
        common.wrong_predictions_num_cores = 0
        common.total_predictions = 0
        common.missed_deadlines = 0

        DTPM_train_model.main("")

        if iter == 0:
            common.first_DAgger_iteration = True
        else:
            common.first_DAgger_iteration = False

        DASH_Sim_v0.run_simulator()

    sim_time = float(float(time.time() - start_time)) / 60.0
    print("--- {:.2f} minutes ---".format(sim_time))

if __name__ == '__main__':
    main()



