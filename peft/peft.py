"""!
@brief internal implementation used for scheduling a task DAG with any of the PEFT family of algorithms
"""

from collections import deque, namedtuple
from math import inf
from peft.gantt import showGanttChart
from types import SimpleNamespace

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger('peft')

ScheduleEvent = namedtuple('ScheduleEvent', 'task start end proc')

"""!
Default computation matrix - taken from Arabnejad 2014 PEFT paper
computation matrix: v x q matrix with v tasks and q PEs
"""
W0 = np.array([
    [22, 21, 36],
    [22, 18, 18],
    [32, 27, 19],
    [7, 10, 17],
    [29, 27, 10],
    [26, 17, 9],
    [14, 25, 11],
    [29, 23, 14],
    [15, 21, 20],
    [13, 16, 16]
])

"""!
Default communication matrix - not listed in Arabnejad 2014 PEFT paper
communication matrix: q x q matrix with q PEs

Note that a communication cost of 0 is used for a given processor to itself
"""
C0 = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

def schedule_dag(dag, computation_matrix=W0, communication_matrix=C0, proc_schedules=None, time_offset=0, relabel_nodes=True):
    """!
    Given an application DAG and a set of matrices specifying PE bandwidth and (task, pe) execution times, computes the PEFT schedule
    of that DAG onto that set of PEs 

    @param dag: a NetworkX DiGraph instance that contains all the tasks to be scheduled as well as their dependency and communication volume relationships
    @param computation_matrix: the matrix of task to PE computation cost estimates used to inform scheduling decisions
    @param communication_matrix: the matrix of PE to PE communication cost estimates used to inform scheduling decisions
    @param proc_schedules: if existing schedules need to be passed in to PEFT (i.e. we are not scheduling onto an idle system), then they can be passed in as fixed constraints here
    @param time_offset: if the schedule here should be generated relative to some time offset rather than the current instance, specify so here
    @param relabel_nodes: if the nodes in the incoming DAG need to be relabeled to not conflict with tasks specified in the proc_schedules constraints, specify so here
    """
    if proc_schedules == None:
        proc_schedules = {}

    _self = {
        'computation_matrix': computation_matrix,
        'communication_matrix': communication_matrix,
        'task_schedules': {},
        'proc_schedules': proc_schedules,
        'numExistingJobs': 0,
        'time_offset': time_offset,
        'root_node': None,
        'optimistic_cost_table': None
    }
    _self = SimpleNamespace(**_self)

    for proc in proc_schedules:
        _self.numExistingJobs = _self.numExistingJobs + len(proc_schedules[proc])

    if relabel_nodes:
        dag = nx.relabel_nodes(dag, dict(map(lambda node: (node, node+_self.numExistingJobs), list(dag.nodes()))))
    else:
        #Negates any offsets that would have been needed had the jobs been relabeled
        _self.numExistingJobs = 0

    for i in range(_self.numExistingJobs + len(_self.computation_matrix)):
        _self.task_schedules[i] = None
    for i in range(len(_self.communication_matrix)):
        if i not in _self.proc_schedules:
            _self.proc_schedules[i] = []

    for proc in proc_schedules:
        for schedule_event in proc_schedules[proc]:
            _self.task_schedules[schedule_event.task] = schedule_event

    # Nodes with no successors cause the any expression to be empty    
    root_node = [node for node in dag.nodes() if not any(True for _ in dag.predecessors(node))]
    assert len(root_node) == 1, f"Expected a single root node, found {len(root_node)}"
    root_node = root_node[0]
    _self.root_node = root_node

    logger.debug(""); logger.debug("====================== Performing Optimistic Cost Table Computation ======================\n"); logger.debug("")
    _self.optimistic_cost_table = _compute_optimistic_cost_table(_self, dag)

    logger.debug(""); logger.debug("====================== Computing EFT for each (task, processor) pair and scheduling in order of decreasing Rank-U ======================"); logger.debug("")
    sorted_nodes = sorted(dag.nodes(), key=lambda node: dag.nodes()[node]['rank'], reverse=True)
    if sorted_nodes[0] != root_node:
        logger.debug("Root node was not the first node in the sorted list. Must be a zero-cost and zero-weight placeholder node. Rearranging it so it is scheduled first\n")
        idx = sorted_nodes.index(root_node)
        sorted_nodes[idx], sorted_nodes[0] = sorted_nodes[0], sorted_nodes[idx]
    logger.debug(f"Scheduling tasks in this order: {sorted_nodes}")
    for node in sorted_nodes:
        if _self.task_schedules[node] is not None:
            continue
        minTaskSchedule = ScheduleEvent(node, inf, inf, -1)
        minOptimisticCost = inf
        for proc in range(len(communication_matrix)):
            taskschedule = _compute_eft(_self, dag, node, proc)
            if (taskschedule.end + _self.optimistic_cost_table[node][proc] < minTaskSchedule.end + minOptimisticCost):
                minTaskSchedule = taskschedule
                minOptimisticCost = _self.optimistic_cost_table[node][proc]
        _self.task_schedules[node] = minTaskSchedule
        _self.proc_schedules[minTaskSchedule.proc].append(minTaskSchedule)
        _self.proc_schedules[minTaskSchedule.proc] = sorted(_self.proc_schedules[minTaskSchedule.proc], key=lambda schedule_event: schedule_event.end)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('\n')
            for proc, jobs in _self.proc_schedules.items():
                logger.debug(f"Processor {proc} has the following jobs:")
                logger.debug(f"\t{jobs}")
            logger.debug('\n')
        for proc in range(len(_self.proc_schedules)):
            for job in range(len(_self.proc_schedules[proc])-1):
                first_job = _self.proc_schedules[proc][job]
                second_job = _self.proc_schedules[proc][job+1]
                assert first_job.end <= second_job.start, \
                f"Jobs on a particular processor must finish before the next can begin, but job {first_job.task} on processor {first_job.proc} ends at {first_job.end} and its successor {second_job.task} starts at {second_job.start}"
    
    dict_output = {}
    for proc_num, proc_tasks in _self.proc_schedules.items():
        for idx, task in enumerate(proc_tasks):
            if idx > 0 and (proc_tasks[idx-1].end - proc_tasks[idx-1].start > 0):
                dict_output[task.task] = (proc_num, idx, [proc_tasks[idx-1].task])
            else:
                dict_output[task.task] = (proc_num, idx, [])

    return _self.proc_schedules, _self.task_schedules, dict_output
    
def _compute_optimistic_cost_table(_self, dag):
    """
    Uses a basic BFS approach to traverse upwards through the graph building the optimistic cost table along the way
    """

    optimistic_cost_table = {}

    terminal_node = [node for node in dag.nodes() if not any(True for _ in dag.successors(node))]
    assert len(terminal_node) == 1, f"Expected a single terminal node, found {len(terminal_node)}"
    terminal_node = terminal_node[0]

    diagonal_mask = np.ones(_self.communication_matrix.shape, dtype=bool)
    np.fill_diagonal(diagonal_mask, 0)
    avgCommunicationCost = np.mean(_self.communication_matrix[diagonal_mask])
    for edge in dag.edges():
        logger.debug(f"Assigning {edge}'s average weight based on average communication cost. {float(dag.get_edge_data(*edge)['weight'])} => {float(dag.get_edge_data(*edge)['weight']) / avgCommunicationCost}")
        nx.set_edge_attributes(dag, { edge: float(dag.get_edge_data(*edge)['weight']) / avgCommunicationCost }, 'avgweight')

    optimistic_cost_table[terminal_node] = _self.computation_matrix.shape[1] * [0]
    dag.nodes()[terminal_node]['rank'] = 0
    visit_queue = deque(dag.predecessors(terminal_node))
    
    node_can_be_processed = lambda node: all(successor in optimistic_cost_table for successor in dag.successors(node))
    while visit_queue:
        node = visit_queue.pop()

        while node_can_be_processed(node) is not True:
            try:
                node2 = visit_queue.pop()
            except IndexError:
                raise RuntimeError(f"Node {node} cannot be processed, and there are no other nodes in the queue to process instead!")
            visit_queue.appendleft(node)
            node = node2

        optimistic_cost_table[node] = _self.computation_matrix.shape[1] * [0]
        logger.debug(f"Computing optimistic cost table entries for node: {node}")

        # Perform OCT kernel
        # Need to build the OCT entries for every task on each processor
        for curr_proc in range(_self.computation_matrix.shape[1]):
            # Need to maximize over all the successor nodes
            max_successor_oct = -inf
            for succnode in dag.successors(node):
                logger.debug(f"\tLooking at successor node: {succnode}")
                # Need to minimize over the costs across each processor
                min_proc_oct = inf
                for succ_proc in range(_self.computation_matrix.shape[1]):
                    successor_oct = optimistic_cost_table[succnode][succ_proc]
                    successor_comp_cost = _self.computation_matrix[succnode][succ_proc]
                    successor_comm_cost = dag[node][succnode]['avgweight'] if curr_proc != succ_proc else 0
                    cost = successor_oct + successor_comp_cost + successor_comm_cost
                    logger.debug(f"If node {node} is on {curr_proc} and successor {succnode} is on {succ_proc}, the optimistic cost entry is {cost}")
                    if cost < min_proc_oct:
                        min_proc_oct = cost
                if min_proc_oct > max_successor_oct:
                    max_successor_oct = min_proc_oct
            assert max_successor_oct != -inf, f"No node should have a maximum successor OCT of {-inf} but {node} does when looking at processor {curr_proc}"
            optimistic_cost_table[node][curr_proc] = max_successor_oct
        # End OCT kernel
        dag.nodes()[node]['rank'] = np.mean(optimistic_cost_table[node])
        visit_queue.extendleft([prednode for prednode in dag.predecessors(node) if prednode not in visit_queue])

    return optimistic_cost_table

def _compute_eft(_self, dag, node, proc):
    """
    Computes the EFT of a particular node if it were scheduled on a particular processor
    It does this by first looking at all predecessor tasks of a particular node and determining the earliest time a task would be ready for execution (ready_time)
    It then looks at the list of tasks scheduled on this particular processor and determines the earliest time (after ready_time) a given node can be inserted into this processor's queue
    """
    ready_time = _self.time_offset
    logger.debug(f"Computing EFT for node {node} on processor {proc}")
    for prednode in list(dag.predecessors(node)):
        predjob = _self.task_schedules[prednode]
        assert predjob != None, f"Predecessor nodes must be scheduled before their children, but node {node} has an unscheduled predecessor of {prednode}"
        logger.debug(f"\tLooking at predecessor node {prednode} with job {predjob} to determine ready time")
        if _self.communication_matrix[predjob.proc, proc] == 0:
            ready_time_t = predjob.end
        else:
            ready_time_t = predjob.end + dag[predjob.task][node]['weight'] / _self.communication_matrix[predjob.proc, proc]
        logger.debug(f"\tNode {prednode} can have its data routed to processor {proc} by time {ready_time_t}")
        if ready_time_t > ready_time:
            ready_time = ready_time_t
    logger.debug(f"\tReady time determined to be {ready_time}")

    computation_time = _self.computation_matrix[node-_self.numExistingJobs, proc]
    job_list = _self.proc_schedules[proc]
    for idx in range(len(job_list)):
        prev_job = job_list[idx]
        if idx == 0:
            if (prev_job.start - computation_time) - ready_time > 0:
                logger.debug(f"Found an insertion slot before the first job {prev_job} on processor {proc}")
                job_start = ready_time
                min_schedule = ScheduleEvent(node, job_start, job_start+computation_time, proc)
                break
        if idx == len(job_list)-1:
            job_start = max(ready_time, prev_job.end)
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
            break
        next_job = job_list[idx+1]
        #Start of next job - computation time == latest we can start in this window
        #Max(ready_time, previous job's end) == earliest we can start in this window
        #If there's space in there, schedule in it
        logger.debug(f"\tLooking to fit a job of length {computation_time} into a slot of size {next_job.start - max(ready_time, prev_job.end)}")
        if (next_job.start - computation_time) - max(ready_time, prev_job.end) >= 0:
            job_start = max(ready_time, prev_job.end)
            logger.debug(f"\tInsertion is feasible. Inserting job with start time {job_start} and end time {job_start + computation_time} into the time slot [{prev_job.end}, {next_job.start}]")
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
            break
    else:
        #For-else loop: the else executes if the for loop exits without break-ing, which in this case means the number of jobs on this processor are 0
        min_schedule = ScheduleEvent(node, ready_time, ready_time + computation_time, proc)
    logger.debug(f"\tFor node {node} on processor {proc}, the EFT is {min_schedule}")
    return min_schedule    

def readCsvToNumpyMatrix(csv_file):
    """!
    Given an input file consisting of a comma separated list of numeric values with a single header row and header column, 
    this function reads that data into a numpy matrix and strips the top row and leftmost column
    """
    with open(csv_file) as fd:
        logger.debug(f"Reading the contents of {csv_file} into a matrix")
        contents = fd.read()
        contentsList = contents.split('\n')
        contentsList = list(map(lambda line: line.split(','), contentsList))
        contentsList = contentsList[0:len(contentsList)-1] if contentsList[len(contentsList)-1] == [''] else contentsList
        
        matrix = np.array(contentsList)
        matrix = np.delete(matrix, 0, 0) # delete the first row (entry 0 along axis 0)
        matrix = np.delete(matrix, 0, 1) # delete the first column (entry 0 along axis 1)
        matrix = matrix.astype(float)
        logger.debug(f"After deleting the first row and column of input data, we are left with this matrix:\n{matrix}")
        return matrix

def readCsvToDict(csv_file):
    """!
    Given an input file consisting of a comma separated list of numeric values with a single header row and header column, 
    this function reads that data into a dictionary with keys that are node numbers and values that are the CSV lists
    """
    with open(csv_file) as fd:
        matrix = readCsvToNumpyMatrix(csv_file)
        
        outputDict = {}
        for row_num, row in enumerate(matrix):
            outputDict[row_num] = row
        return outputDict

def readDagMatrix(dag_file, show_dag=False):
    """!
    Given an input file consisting of a connectivity matrix, reads and parses it into a networkx Directional Graph (DiGraph)
    """
    matrix = readCsvToNumpyMatrix(dag_file)

    dag = nx.DiGraph(matrix)
    dag.remove_edges_from(
        # Remove all edges with weight of 0 since we have no placeholder for "this edge doesn't exist" in the input file
        [edge for edge in dag.edges() if dag.get_edge_data(*edge)['weight'] == '0.0']
    )

    if show_dag:
        nx.draw(dag, pos=nx.nx_pydot.graphviz_layout(dag, prog='dot'), with_labels=True)
        plt.show()

    return dag

def generate_argparser():
    """!
    For when this peft library is being invoked as a standalone commandline application, generate an argparser
    """
    parser = argparse.ArgumentParser(description="A tool for finding PEFT schedules for given DAG task graphs")
    parser.add_argument("-d", "--dag_file", 
                        help="File containing input DAG to be scheduled. Uses default 10 node dag from Arabnejad 2014 if none given.",
                        type=str, default="test/peftgraph_task_connectivity.csv")
    parser.add_argument("-p", "--pe_connectivity_file", 
                        help="File containing connectivity/bandwidth information about PEs. Uses a default 3x3 matrix from Arabnejad 2014 if none given.",
                        type=str, default="test/peftgraph_resource_BW.csv")
    parser.add_argument("-t", "--task_execution_file", 
                        help="File containing execution times of each task on each particular PE. Uses a default 10x3 matrix from Arabnejad 2014 if none given.",
                        type=str, default="test/peftgraph_task_exe_time.csv")
    parser.add_argument("-l", "--loglevel", 
                        help="The log level to be used in this module. Default: INFO", 
                        type=str, default="INFO", dest="loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--showDAG", 
                        help="Switch used to enable display of the incoming task DAG", 
                        dest="showDAG", action="store_true")
    parser.add_argument("--showGantt", 
                        help="Switch used to enable display of the final scheduled Gantt chart", 
                        dest="showGantt", action="store_true")
    return parser

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    logger.setLevel(logging.getLevelName(args.loglevel))
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.getLevelName(args.loglevel))
    consolehandler.setFormatter(logging.Formatter("%(levelname)8s : %(name)16s : %(message)s"))
    logger.addHandler(consolehandler)

    communication_matrix = readCsvToNumpyMatrix(args.pe_connectivity_file)
    computation_matrix = readCsvToNumpyMatrix(args.task_execution_file)
    dag = readDagMatrix(args.dag_file, args.showDAG) 

    processor_schedules, _, _ = schedule_dag(dag, communication_matrix=communication_matrix, computation_matrix=computation_matrix)
    for proc, jobs in processor_schedules.items():
        logger.info(f"Processor {proc} has the following jobs:")
        logger.info(f"\t{jobs}")
    if args.showGantt:
        showGanttChart(processor_schedules)
