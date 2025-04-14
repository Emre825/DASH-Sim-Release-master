import re, os, glob, sys
import matplotlib

dir = sys.argv[1]

max_dagger_num = 0
files = glob.glob(dir + '/reports/*dagger*')
for file in files :
    num_lines  = os.popen('wc -l ' + file).read().strip().split(' ')[0]
    num_lines  = int(num_lines)
    if num_lines < 10 :
        continue

    matchobj   = re.match(r'.*report(.*)_dagger(.*).rpt', file)
    scale      = matchobj.group(1)
    dagger_num = int(matchobj.group(2))
    if dagger_num > max_dagger_num :
        max_dagger_num = dagger_num
    ## if dagger_num > max_dagger_iter :
## for file in files :

scales = [
'1000-1001-1' ,
'750-751-1' ,
'600-601-1' ,
'500-501-1' ,
'400-401-1' ,
'300-301-1' ,
'250-251-1' ,
'200-201-1' ,
'175-176-1' ,
'150-151-1' ,
'100-101-1' ,
'80-81-1' ,
'65-66-1' ,
'50-51-1' ,
'40-41-1' ,
'30-31-1' ,
'20-21-1' ,
]

exectime_d = dict()
dagger_iter_list = []

# iteration_check_string = 'dagger' + str(max_dagger_iter + 1)
# files = [file for file in files if iteration_check_string not in file]

for scale in scales :
    for dagger_num in range(max_dagger_num) :
        dagger_num = dagger_num + 1
        matchobj   = re.match(r'.*report(.*)_dagger(.*).rpt', file)
        file = dir + '/reports/' + 'report_ETF_IL_' + scale + '_dagger' + str(dagger_num) + '.rpt'
        fp = open(file, 'r')

        for line in fp :
            line = line.strip()
            matchobj = re.match(r'.*Completed.*injection rate:(.*), conc.*ave_execution_time:(.*)', line)
            if matchobj :
                inj_rate = matchobj.group(1)
                exec_time = matchobj.group(2)
                # energy    = matchobj.group(3)
                
                if not dagger_num in exectime_d :
                    # exectime_d[dagger_num] = [[inj_rate, exec_time, energy]]
                    exectime_d[dagger_num] = [[inj_rate, exec_time]]
                else :
                    # exectime_d[dagger_num].append([inj_rate, exec_time, energy])
                    exectime_d[dagger_num].append([inj_rate, exec_time])
                    dagger_iter_list.append(int(dagger_num))

# max_dagger_iter = max(dagger_iter_list)

length = len(exectime_d[1])
for i in range(length) :
    for dagger_iter in range(max_dagger_num) :

        dagger_iter = int(dagger_iter) + 1
        # sorted_data = sorted(exectime_d[str(dagger_iter)]) #, key=lambda x:x[1])
       
        # print(str(dagger_iter) + ' ', end='')
        # print('%.4f %.2f %.4f ' %(float(exectime_d[dagger_iter][i][0]), float(exectime_d[dagger_iter][i][1]), float(exectime_d[dagger_iter][i][2])), end='')
        print('%.4f %.2f ' %(float(exectime_d[dagger_iter][i][0])*1000, float(exectime_d[dagger_iter][i][1])), end='')
    print()

