@page details DASH-Sim in Depth
@tableofcontents


@section config Configuring DASH-Sim

DASH-Sim configured via config_file.ini. The file is composed of six sections:
    - *DEFAULT* is the section where all main parameters for a simulation are assigned. \n
    To run a simulation:
        -# an @ref soc,
        -# an @ref application or **application configuration files**, and 
        -# a **scheduler** must be defined.
    - *HEFT SCHEDULER* is the section where all HEFT related parameters are assigned.
    - *IL SCHEDULER* is the section where all Imitation Learning based (IL) scheduler parameters are assigned.
    - *TRACE* is the section where all parameters related to tracing are assigned.
    - *POWER MANAGEMENT* is the section where analytical power and temperature model parameters are assigned.
    - *SIMULATION MODE* is the section where simulation mode is defined and related parameters are assigned.
    - *COMMUNICATION MODE* is the section where communication mode is defined and related parameters are assigned.
    - *DEBUG* is the section where various debug messages can be enabled and/or disabled.
    - *INFO* is the section where various information messages can be enabled and/or disabled.
       
@section soc SoC configuration file
DASH-Sim enables instantiating a wide range of SoC configurations with different types of general- and special purpose
processing elements (PEs) via an SoC configuration file. \n
A list of PEs and its characteristics are provided in the file are stored in the resource database


<table>
<caption id="resource database">List of PE attributes in resource database</caption>
<tr><th>Attribute        <th>Description
<tr><td> Type             <td> Defines type of PE, Example: CPU, accelerator etc.
<tr><td> Capacity             <td> Number of simultaneous threads a PE can execute
<tr><td> DVFS policy             <td> Policy which controls PE frequency and voltage at runtime
<tr><td> Operating performance point             <td> Operating frequencies and corresponding voltages for a PE
<tr><td> Execution time profile            <td> Defines the execution time of supported tasks on each PE
<tr><td> Power consumption profile             <td> Provides the power consumption profile of each PE
</table>

The configuration file is in a text format and the structure is given below.

\code{.py}
#Configuration file of the Resources available in DASH-SoC
#Lines that start with "#" are comments
#
#Add a new resource using the keyword: add_new_resource
#Always add MEMORY last as a special resource, which will be used for communication and does not support any functionality
#The following lines must have the attributes below
#The format: add_new_resource $resource_type (string)  $resource_name (string) $resource_id (int) $capacity (int) $num_of_supported_functionality (int) $DVFS_mode (string)
#             $functionality_name (string) $execution_time (float)
#Note: for type, use the following abbreviations
#       central processing unit --> CPU
#       Arm LITTLE cluster      --> LTL
#       Arm big cluster         --> BIG
#       accelerator             --> ACC
#       memory                  --> MEM
#
#opp $frequency (int - MHz) $voltage (int - mV), defines the Operating Performance Points (OPPs) with frequency and voltage tuples
#
#trip_freq $trip_1 $trip_2 $trip_3 ..., defines the frequencies that are set at each trip point if throttling is enabled. "-1" means that the frequency is not modified#
#
#power_profile $frequency $power_1 $power_2 $power_3 ... $power_max_capacity
#e.g., power_profile 1000 0.19 0.29 0.35 0.40. At 1GHz, the power for 1 core is 0.19W, 2 cores is 0.29W, and so on.
#
#PG_profile $frequency $power_1 $power_2 $power_3 ... $power_max_capacity
#
#Example: The following lines add a new CPU with name=P1, ID=0, capacity=1 and that can run 3 different tasks using "performance" DVFS mode
#
#          add_new_resource CPU P1 0 1 3 performance
#          opp 1000 1150
#          trip_freq -1 -1 -1
#          power_profile 1000 0.1
#          PG_profile 1000 0.1
#          scrambler 12
#          reed_solomon_encoder 15
#          bpsk_modulation 18
#
#After adding resources, use keyword comm_band to add communication bandwidth between resources
#The format: comm_band $source_id (int) $destination_id (int) bandwidth (int)
#Example: The following line adds communication bandwidth between resource 0 (source)
#          and resource 1 (destination)
#                    
#          comm_band 0 1 5
\endcode

To see an example of an SoC configuration file please see one of the **SoC.XXX.txt** files

@section application Application configuration file
In DASH-Sim applications are modeled as a directed acyclic graph (DAG). In the graph nodes represent a task and edges shows data dependency between the tasks.\n
For example, the left of the figure below depicts the block diagrams of WiFi transmitter (WiFi-TX) and receiver (WiFi-RX) applications. On the right, however, \n
the corresponding DAGs for these two applications used in DASH-Sim presented. These reference design of WiFi-TX and WiFi-RX applcaitions contain five parallel \n
chains of tasks as seen in the figure below.


@image html WiFi_BD_and_DAG.png (a) Block diagrams and (b) DAGs for WiFi-TX and WiFi-RX applications. height=55% width=55% 

The configuration file is in a text format and the structure is given below.

\code{.py}
# Configuration file of the Temporal Mitigation
# Lines that start with "#" are comments
# Start with specifying the job name with keyword: job_name
# Then, in the next line, add tasks using the keyword: add_new_tasks
# After that the following three lines must have the attributes below
# $task_name (string) $task_id (int) $task_predecessors (list)
# $task_name (string) HEAD (TAIL), if the task is the head or tail of the task graph
# $task_name (string) $earliest_start (int) $deadline (int) $input_vol (int - bits) $output_vol (int - bits)
# The format: add_new_tasks $num_of_tasks (int)
#             $task_name (string) $task_id (int) $task_predecessors (list)
#             $task_name (string) HEAD (string)
#             $task_name (string) $earliest_start (int) $deadline (int) $input_vol (int) $output_vol (int)
# Example: The following lines add a new task with ID=0, and
#          there is no predecessor for this task
#          (empty list means there is no dependency)
#          This task is the head of the task graph
#          earliest start time and deadline are 0 and 10, respectively
#          input and output volumes are 1024 bits
#
#          add_new_tasks 1
#          scrambler 0
#          scrambler HEAD
#          scrambler earliest_start 0 deadline 10 input_vol 1024 output_vol 1024
\endcode

To see an example of an Application configuration file please see one of the **job_XXX.txt** files

@section scheduler Built-in schedulers

DASH-Sim facilitates plug-and-play simulation of scheduling algorithms; it also incorporates built-in heuristic and table-based schedulers to aid developers and \n
provide a baseline for users. DASH-Sim also includes power dissipation and thermal models that enable users to design and evaluate new dynamic thermal and power \n 
management (%DTPM) policies. The latter, however, will be discussed in another section.

Some of the scheduling algorithms are:
    - **Minimum Execution Time (MET) scheduler** assigns a ready task to a PE that achieves the minimum expected execution time following a FIFO policy (<a href=http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.25.4721&rep=rep1&type=pdf>T. D. Braun et al.</a>).
    - **Earliest Task First (ETF) scheduler** utilizes the information about the communication cost between tasks and the current status of all PEs to make a scheduling decision (<a href=https://ieeexplore.ieee.org/abstract/document/1558639>J. Blythe et al.</a>).\n
    Consider a set of PEs \f$ P = \{p_0, p_1, ..., p_{q-1}\} \f$ and a task graph with tasks \f$N = \left\{n_0, n_1, ..., n_{k-1}\right\}\f$. For every task and PE pair, we define the following quantities:
        - *Estimated Execution Time, EET(n; p)*, is the total amount of time that the resource *p* takes to run the task *n*.
        - *Estimated Availability Time, EAT(n; p)*, is defined as the time when the resource *p* will finish executing all tasks in its queue before the task *n*.
        - *Data Availability Time, DAT(n; p)*, is the earliest time at which all required data becomes available at resource *p* to perform task *n*.
        - *Estimated Completion Time, ECT(n; p)*, is the completion time of task n executed on resource *p*. It is given by: \n \f$ECT(n,p) = EET(n,p)+max[EAT(n,p),DAT(n,p)]\f$ \n
    The ETF scheduler finds the resource with the minimum ECT value for each available task. Next, the task with the minimum ECT value is determined and it is \n 
    scheduled first. This routine is performed until all available tasks are assigned to a PE.
    
    - **Constraint Programming scheduler** is formulated using IBM ILOG CPLEX Optimization Studio (<a href=https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.studio.help/pdf/usrcplex.pdf>CPLEX's User Manual</a>) to obtain an optimal schedule whenever the problem \n
    size allows. The simulation framework dynamically calls the CP solver at the injection of each frame (i.e., an instance of an application) to find the schedule as a function \n 
    of the current system state. Then, the obtained schedule is stored in a lookup table and tasks are assigned to PEs accordingly as the simulation proceeds. To see implementation \n
    details are in CP_models.py
    
Users can implement their own scheduling algorithms as a member function in Scheduler class.

@section dtpm Built-in DTPM policies

State-of-the-art SoCs support multiple voltage-frequency domains and DVFS, which enables users to optimize for various power-performance trade-offs. To support this capability, \n 
DS3 allows each PE to have a range of OPPs, configurable in the resource database. The OPPs are voltage-frequency tuples that represent all supported frequencies of a given PE, \n 
which can be exploited by %DTPM algorithms to tune the SoC at runtime.

The integrated power and temperature models enable user to implement a wide range of %DTPM policies using DASH-Sim. To provide a solid baseline to users, DASH-Sim provides built-in \n
DVFS policies that are commonly used in commercial SoCs. More specifically, users use the input configuration file to set the %DTPM policy to <a href=https://www.kernel.org/doc/Documentation/cpu-freq/governors.txt>ondemand, performance and powersave</a> \n 
governors, or to a custom %DTPM governor.
    - **Ondemand Governor**: The ondemand governor controls the OPP of each PE as a function of its utilization. The supported voltage-frequency pairs of a PE are given \n
    by the following set: 
    \n \f$\mathcal{OPP} = \{(V_1, f_1), (V_2, f_2), \ldots, (V_k,f_k)\}\f$ \n
    where k is the number of operating points supported by that PE. Suppose that the PE currently operates at \f$(V2; f2)\f$. If the utilization of the PE is less than a \n 
    user-defined threshold, then the *ondemand governor* decreases the frequency and voltage such that the new OPP becomes \f$(V1; f1)\f$. If the utilization is greater \n 
    than another user-defined threshold, the OPP is increased to the maximum frequency. Otherwise, the OPP stays at the current value, i.e., \f$(V2; f2)\f$.
    - **Performance Governor**: This policy sets the frequency and voltage of all PEs to their maximum values to minimize execution time.
    - **Powersave Governor**: This policy sets the frequency and voltage of all PEs to their minimum values to minimize power consumption.
    - **User-Specified Values**: This policy enables users to set the OPP (i.e., frequency and voltage) of each PE individually to a constant value within the permitted range. \n 
    It enables thorough power-performance exploration by sweeping the OPPs.

Users can also implement their own %DTPM algorithms as a member function in DTPM class.

