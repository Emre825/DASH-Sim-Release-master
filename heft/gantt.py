"""!
Basic implementation of Gantt chart plotting using Matplotlib
Taken from https://sukhbinder.wordpress.com/2016/05/10/quick-gantt-chart-with-matplotlib/ and adapted as necessary (i.e. removed Date logic, etc)
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np

def showGanttChart(proc_schedules):
    """!
    Given a dictionary of processor-task schedules, displays a Gantt chart generated using Matplotlib
    @param proc_schedules: a map containing keys of each processor and has values that are a list of heft.ScheduleEvent-based tasks
    """  
    
    processors = list(proc_schedules.keys())

    color_choices = ['red', 'blue', 'green', 'cyan', 'magenta']

    ilen=len(processors)
    pos = np.arange(0.5,ilen*0.5+0.5,0.5)
    fig = plt.figure(figsize=(15,6))
    ax = fig.add_subplot(111)
    for idx, proc in enumerate(processors):
        for job in proc_schedules[proc]:
            ax.barh((idx*0.5)+0.5, job.end - job.start, left=job.start, height=0.3, align='center', edgecolor='black', color='white', alpha=0.95)
            ax.text(0.5 * (job.start + job.end - len(str(job.task))-0.25), (idx*0.5)+0.5 - 0.03125, job.task, color=color_choices[((job.task) // 10) % 5], fontweight='bold', fontsize=18, alpha=0.75)
    
    locsy, labelsy = plt.yticks(pos, processors)
    plt.ylabel('Processor', fontsize=16)
    plt.xlabel('Time', fontsize=16)
    plt.title('Standalone HEFT', fontsize=16)
    plt.setp(labelsy, fontsize = 14)
    ax.set_ylim(bottom = -0.1, top = ilen*0.5+0.5)
    ax.set_xlim(left = -5)
    ax.grid(color = 'g', linestyle = ':', alpha=0.5)

    plt.show()