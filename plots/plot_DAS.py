import os, re, glob, sys, numpy as np
import matplotlib.pyplot as plt

## Specify the schedulers to plot
schedulers = ['ETF_DAS', 'dataset_DAS', 'policy_DAS']

## Initialize figure
plt.rcParams.update({'font.size': 18})
figure = plt.figure(figsize=(10,6))

## Iterate through schedulers and plot injection rate vs execution time
for scheduler in schedulers :
    
    ## Get list of all report files
    files = glob.glob('./reports/report_' + scheduler + '_*.rpt')
    
    ## Initialize lists to store injection rates and execution times
    inj_rates = []
    exe_times = []
    data_rates = []
    scales = []
    
    ## Iterate through files to get injection rates and execution times
    for file in files :
        #matchobj = re.match(r'./reports/report_*_(.*).rpt', file)
        file_handle = open(file, 'r')
        for line in file_handle :
            line = line.strip()
            if 'Completed all' in line :
                matchobj = re.match(r'.*Completed.* scale = (.*), injection rate:(.*), conc.*execution_time:(.*), ave_throughput:(.*), EDP.*', line)
                scale = int(matchobj.group(1))
                inj_rate = float(matchobj.group(2))
                exe_time = float(matchobj.group(3))
                data_rate = float(matchobj.group(4))
                scales.append(scale)
                inj_rates.append(inj_rate)
                exe_times.append(exe_time)
                data_rates.append(data_rate)
        ## for line in file_handle :
        file_handle.close()
    ## for file in files :
    
    ## Sort injection rates
    scales = np.array(scales)
    inj_rates = np.array(inj_rates)
    exe_times = np.array(exe_times)
    data_rates = np.array(data_rates)
    indices = np.argsort(scales)
    #isdr = np.sort(data_rates)
    iset = exe_times[indices]
    isdr = data_rates[indices]
    
    if scheduler == 'ETF_DAS' :
        plt.plot(isdr, iset, color='black', label=scheduler, marker='o', markerfacecolor='white', markersize=7)
    elif scheduler == 'dataset_DAS' :
        plt.plot(isdr, iset, color='blue', label=scheduler, marker='^', markerfacecolor='blue', markersize=7)
    else:
        plt.plot(isdr, iset, color='red', label=scheduler, marker='*', markerfacecolor='white', markersize=7, linestyle='dashed')

plt.xlabel('Data Rate (Mbps)')
plt.ylabel('Average Execution Time (us)')
plt.legend(['ETF', 'LUT', 'DAS'])
plt.grid(linestyle=':')
plt.show()
