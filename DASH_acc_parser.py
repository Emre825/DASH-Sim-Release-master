'''!
@brief This file contains the code to parse DASH-SoC accelerators given in ACC_model.csv file.
'''

import pandas as pd

import common

def acc_parse():
    '''!
    Read and parse the configurations of all accelerators.
    '''
    csv_file = pd.read_csv('ACC_model.csv')
    for index, entry in csv_file.iterrows():
        new_acc                         = common.ResourceAcc()
        new_acc.type                    = entry['PE Type']                      # PE type of the accelerator (FIR,DAP,FFT,..)
        new_acc.config                  = entry['Config']                       # Configuration type of accelerator for the executed task
        new_acc.programming_latency     = entry['Programming Latency']          # Programming latency if task is being executed for the first time
        new_acc.leakage_power           = entry['Leakage Power (W)']            # Leakage power consumption of the task
        new_acc.dynamic_power           = entry['Dynamic Power (W)']            # Dynamic power consumption of the task
        new_acc.DAP_subPEs              = entry['DAP sub-PE']                   # How many sub-PEs of DAP should be utilized
        dict_entry = {new_acc.type + ',' + new_acc.config : new_acc}
        common.resource_matrix_Acc.dict.update(dict_entry)
