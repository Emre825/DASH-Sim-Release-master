import os, re, glob, sys, numpy as np

# scheduler = sys.argv[1]
scheduler = 'ETF_Oracle'
#scheduler = 'ETF_IL'

files = glob.glob('./reports/report_' + scheduler + '_*1.rpt')

inj_rates = []
exe_times = []

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

inj_rates = np.array(inj_rates)
exe_times = np.array(exe_times)

indices = np.argsort(inj_rates)
sorted_inj_rates = np.sort(inj_rates)
sorted_exe_times = exe_times[indices]

for index, inj_rate in enumerate(sorted_inj_rates) :
    print('%.4f %.4f' %(inj_rate, sorted_exe_times[index]))
