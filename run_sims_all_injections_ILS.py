import DASH_Sim_v0 as ds3
import multiprocessing
import platform

## Get current platform type
current_platform = platform.system()

## Specify injection rates to execute
## A scale value of 20 implies that a new application/job/DAG is injected every 20 microseconds
scale_values_list = \
[
'20-21-1' ,
'30-31-1' ,
'40-41-1' ,
'50-51-1' ,
'65-66-1' ,
'80-81-1' ,
'100-101-1' ,
'150-151-1' ,
'175-176-1' ,
'200-201-1' ,
'250-251-1' ,
'300-301-1' ,
'400-401-1' ,
'500-501-1' ,
'600-601-1' ,
'750-751-1' ,
'1000-1001-1' ,
]

## Run with multiprocessing library if platform is Linux
if current_platform.lower() == "linux" :
    p = multiprocessing.Pool(50)
    p.map(ds3.run_simulator, scale_values_list)
else :
    ## Run for all injection rates one-by-one
    for scale_value in scale_values_list :
        ds3.run_simulator(scale_value)
    ## for scale_value in scale_values_list :
## if current_platform.lower() == "linux" :
