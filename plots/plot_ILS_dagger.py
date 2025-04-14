import os, re, glob, sys, numpy as np
import matplotlib.pyplot as plt

## Specify the schedulers to plot
schedulers = ['ETF_Oracle', 'ETF_IL']

## Display
print()
print('Difference in average execution time (us) between Oracle and IL policies (baseline and DAgger iterations)')

## Function to get the difference between two curves
def diff_between_curves(x1, y1, x2, y2) :
    
    ## Convert to numpy arrays
    x1 = np.array(x1)
    y1 = np.array(y1)
    x2 = np.array(x2)
    y2 = np.array(y2)

    ## Fit polynomial to curve-1
    p1 = np.polyfit(x1, y1, 5)
    ## Fit polynomial to curve-2
    p2 = np.polyfit(x2, y2, 5)

    ## Get min and max ranges
    min_x1 = np.min(x1)
    min_x2 = np.min(x2)
    max_x1 = np.max(x1)
    max_x2 = np.max(x2)

    x_min  = max(min_x1, min_x2)
    x_max  = min(max_x1, max_x2)
 
    x = x_min
    xlist = []
    y1_list = []
    y2_list = []

    error = 0
    count = 0
    while x < x_max :
        v1 = np.polyval(p1, x)
        v2 = np.polyval(p2, x)

        xlist.append(x)
        y1_list.append(v1)
        y2_list.append(v2)
        error += np.abs(v2/v1)
        
        x += 0.1
        count += 1
    ## while x < x_max :
    
    error = error / count
    return error

## def diff_between_curves(x1, x2, y1, y2) :

## Initialize plt
plt.rcParams.update({'font.size': 18})
figure = plt.figure(figsize=(10,8))
legend_list = []

## Initialize lists
oracle_inj_rates = []
oracle_exe_times = []

## Iterate through schedulers and plot injection rate vs execution time
for scheduler in schedulers :
    
    ## Get list of all report files
    files = glob.glob('./reports/report_' + scheduler + '_*-1.rpt')
    
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
                inj_rate = float(matchobj.group(1)) * 1000
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
    isir = inj_rates[indices]
    iset = exe_times[indices]
    
    if scheduler == 'ETF_Oracle' :
        plt.plot(isir, iset, color='black', label=scheduler, marker='o', markerfacecolor='white', markersize=7, linestyle='dashed')
        legend_list.append('Oracle')
        oracle_inj_rates = isir
        oracle_exe_times = iset
    else :
        plt.plot(isir, iset, color='blue', label=scheduler, marker='^', markerfacecolor='blue', markersize=7)
        legend_list.append('IL')
        error = diff_between_curves(oracle_inj_rates, oracle_exe_times, isir, iset)
        print('%15s: %.4f%%' %('IL(Baseline)', error))

## Initialize maximum DAgger iteration number
max_dagger_num = 0

## Get all report files and get maximum DAgger iteration
files = glob.glob('./reports/*dagger*')
for file in files :
    num_lines  = len(open(file).readlines())
    if num_lines < 10 :
        continue

    matchobj   = re.match(r'.*report(.*)_dagger(.*).rpt', file)
    scale      = matchobj.group(1)
    dagger_num = int(matchobj.group(2))
    if dagger_num > max_dagger_num :
        max_dagger_num = dagger_num
    ## if dagger_num > max_dagger_iter :
## for file in files :

## Iterate through reports for each DAgger iteration one by one
for dagger_num in range(max_dagger_num) :
    dagger_num = dagger_num + 1
    files = glob.glob('./reports/report_ETF_IL_*' + str(dagger_num) + '.rpt')

    ## Initialize lists to store injection rates and execution times
    inj_rates = []
    exe_times = []
    
    ## Iterate through reports for each DAgger iteration
    for file in files :
        fp = open(file, 'r')

        ## Parse through the report file
        for line in fp :
            line = line.strip()
            matchobj = re.match(r'.*Completed.*injection rate:(.*), conc.*ave_execution_time:(.*)', line)
            if matchobj :
                inj_rate = float(matchobj.group(1)) * 1000
                exe_time = float(matchobj.group(2))
                inj_rates.append(inj_rate)
                exe_times.append(exe_time)
            ## if matchobj :
        ## for line in fp :
        fp.close()
    ## for file in files :

    ## Sort injection rates
    inj_rates = np.array(inj_rates)
    exe_times = np.array(exe_times)
    indices = np.argsort(inj_rates)
    isir = inj_rates[indices]
    iset = exe_times[indices]
    
    ## Get difference between Oracle and current DAgger iteration
    error = diff_between_curves(oracle_inj_rates, oracle_exe_times, isir, iset)
    print('%15s: %.4f%%' %('IL-DAgger' + str(dagger_num), error))
    
    ## Plot lines
    plt.plot(isir, iset, label='IL-DAgger' + str(dagger_num), marker='x', markersize=7)
    legend_list.append('IL-DAgger-' + str(dagger_num))
## for dagger_num in range(max_dagger_num) :
    
## for dagger_num in range(max_dagger_num) :
plt.xlabel('Injection Rate (jobs/ms)')
plt.ylabel('Average Execution Time (us)')
plt.legend(legend_list, ncol=3, loc='upper center', bbox_to_anchor=(0.5,1.4))
plt.grid(zorder=0, color='gray', linestyle='--', linewidth=0.5)
plt.tight_layout(pad=0.05)
plt.show()

