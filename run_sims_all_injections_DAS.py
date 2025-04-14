import DASH_Sim_v0 as ds3
import multiprocessing
import platform

## Specify parameters to execute
#ds3.common.das_dataset = True
#ds3.common.das_complex_only = False
#ds3.common.das_policy = False

# choose your scale list from excel sheet
scale_values_list = \
[
'25000-25001-1',
'10000-10001-1',
'5000-5001-1',
'4000-4001-1',
'3000-3001-1',
'2500-2501-1',
'2000-2001-1',
'1750-1751-1',
'1000-1001-1',
'500-501-1',
'250-251-1'
]

## Main function
if __name__ == '__main__' :

  ## Get current platform type
  current_platform = platform.system()
  
  ## Run with multiprocessing library if platform is Linux
  if current_platform.lower() == "linux" :
      p = multiprocessing.Pool(14)
      p.map(ds3.run_simulator, scale_values_list)
  else :
      ## Run for all injection rates one-by-one
      for scale_value in scale_values_list :
          ds3.run_simulator(scale_value)
      ## for scale_value in scale_values_list :
  ## if current_platform.lower() == "linux" :
