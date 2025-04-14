@page firstexample First Example
@tableofcontents

For the first example in DASH-Sim, one of the most commonly used canonical task graphs (<a href=https://ieeexplore.ieee.org/document/993206>H. Topcuoglu, S. Hariri, and M.-Y. Wu</a>) is utilized here. \n 
Since this task graph (shown in the figure below) is considered as reference for many list-scheduling studies, it serves as a representative example \n
before analyzing the results from real-world applications.

@image html HEFT_DAG.png A canonical task flow graph with ten tasks. height=30% width=30% 

Each node in this graph represents a task and each edge represents average communication cost across the available pool of PEs for the pair of nodes \n 
sharing that edge. The computation cost table on the right indicates the execution time on each of the PEs. Then a schedule for this task graph can be \n 
obtained from DASH-Sim using the built-in schedulers.

@image html 1st_example.png (a) DAG representation and (b) schedules obtained from DASH-Sim. height=55% width=55% 


@section first_config Configuration file to run first example

The following setup for the config_file.ini is used to generate above schedules.
To have all there schedules, for each run, simply change **scheduler** to:
    - MET
    - ETF
    - CP
    
Configuration file: config_file_first_example.ini

