import os, re, glob, sys, numpy as np
import matplotlib.pyplot as plt

## Specify the schedulers to plot
schedulers = ['ETF_Oracle', 'ETF_IL']

## Initialize figure
plt.rcParams.update({'font.size': 18})
figure = plt.figure(figsize=(10,6))

## Iterate through schedulers and plot injection rate vs execution time
for scheduler in schedulers :
    
    ## Get list of all report files
    files = glob.glob('./reports/report_' + scheduler + '_*1.rpt')
    
    ## Initialize lists to store injection rates and execution times
    inj_rates = []
    exe_times = []
    
    ## Iterate through files to get injection rates and execution times
    for file in files :
        matchobj = re.match(r'./reports/report_*_(.*).rpt', file)
        file_handle = open(file, 'r')
        for line in file_handle :
            line = line.strip()
            if 'Completed all' in line :
                matchobj = re.match(r'.*Completed.*injection rate:(.*), conc.*execution_time:(.*)', line)
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
    isir = np.sort(inj_rates)
    iset = exe_times[indices]
    
    if scheduler == 'ETF_Oracle' :
        plt.plot(isir, iset, color='black', label=scheduler, marker='o', markerfacecolor='white', markersize=7, linestyle='dashed')
    else :
        plt.plot(isir, iset, color='blue', label=scheduler, marker='^', markerfacecolor='blue', markersize=7)
    
#    for index, inj_rate in enumerate(sorted_inj_rates) :
#        print('%.4f %.4f' %(inj_rate, sorted_exe_times[index]))

plt.xlabel('Injection Rate (jobs/ms)')
plt.ylabel('Average Execution Time (us)')
plt.legend(['Oracle', 'IL'])
plt.grid(linestyle=':')
plt.show()
