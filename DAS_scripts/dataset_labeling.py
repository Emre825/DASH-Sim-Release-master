#!/usr/bin/python

import os, re, glob, sys
import numpy as np

sys.path.append(os.getcwd())
# import run_sims_all_injections_ILS

schedulers = ['dataset_DAS', 'ETF_DAS']
## Iterate through schedulers and plot injection rate vs execution time
inj_rate_dict = {}
exe_time_dict = {}
for scheduler in schedulers :
    
    ## Get list of all report files
    files = glob.glob('./reports/report_' + scheduler + '_*.rpt')
    
    ## Initialize lists to store injection rates and execution times
    inj_rates = []
    exe_times = []
    
    ## Iterate through files to get injection rates and execution times
    for file in files :
        #matchobj = re.match(r'./reports/report_*_(.*).rpt', file)
        file_handle = open(file, 'r')
        for line in file_handle :
            line = line.strip()
            if 'Completed all' in line :
                matchobj = re.match(r'.*Completed.*injection rate:(.*), conc.*execution_time:(.*), ave_th.*', line)
                inj_rate = float(matchobj.group(1))
                exe_time = float(matchobj.group(2))
                inj_rates.append(inj_rate)
                exe_times.append(exe_time)
        ## for line in file_handle :
        file_handle.close()
    ## for file in files :
    
    ## Sort injection rates
    inj_rates = np.array(inj_rates)
    exe_times = np.array(exe_times)
    indices = np.argsort(inj_rates)
    sorted_inj_rate = np.sort(inj_rates)
    sorted_exe_time = exe_times[indices]
    inj_rate_dict[scheduler] = sorted_inj_rate
    exe_time_dict[scheduler] = sorted_exe_time

for ind, exetime in enumerate(exe_time_dict[schedulers[0]]):
    if exetime > exe_time_dict[schedulers[1]][ind]:
        crossover_point = (inj_rate_dict[schedulers[0]][ind-1] + inj_rate_dict[schedulers[0]][ind]) / 2
        break
# crossover_point = 584.1
#models = ['25000','10000','5000','3750','2500','2000','1500','1000','500','250','100','20']
models = ['clustera']
folder = ''
## Iterate through all the clusters
for model in models :

    # scale_values_list = run_sims_all_injections_ILS.scale_values_list

    #files = []
    # for scale in scale_values_list :
    # values = scale.split('-')
    # scale = values[0]
    #filename = '../datasets/DAS/' + folder + '/' + folder + '.csv'
    #files.append(filename)
    files = glob.glob('./datasets/DAS/' + folder + '/*.csv')

    output_filename = './datasets/DAS/postprocessed/data_DAS_merged_all_postprocessed' + folder + '.csv'
    output_filehandle = open(output_filename, 'w')

    header = 0

    ## Iterate through all files
    for file in files :
        file_handle = open(file, 'r')

        ## Parse each file 
        for line in file_handle :
            if 'Time' in line:
                if header == 0:
                    output_filehandle.write(line)
                    header = 1
                    continue
                else:
                    continue
            current_line = line.split(",")
            if current_line[-1] == '-9\n':
                if float(current_line[-3]) > crossover_point:
                    new_line = line[:-3] + '1' + line[-1:]
                else:
                    new_line = line[:-3] + '0' + line[-1:]
            else:
                new_line = line
            output_filehandle.write(new_line)
        ## for line in file_handle :
        #header = 0
        file_handle.close()
    ## for file in files :
    output_filehandle.close()


