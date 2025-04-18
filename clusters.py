'''!
@brief This file contains the information about the clusters in the SoC.
'''

class Cluster:
    '''!
    The Cluster class will help manage the power of the SoC
    '''
    def __init__(self, name, ID, type):
        self.name                       = name  # Cluster name
        self.ID                         = ID    # Cluster ID
        self.type                       = type  # Cluster type
        self.PE_list                    = []    # List with the PEs that are comprised in this cluster
        self.DVFS                       = ''    # DVFS mode
        self.power_profile              = {}    # Dict of power values related to the capacity of the cluster for each frequency threshold. e.g., 4 cores, then [P_1core, P_2cores, P_3cores, P_4cores]
        self.PG_profile                 = {}    # Dict of power values when applying PG related to the capacity of the cluster for each frequency threshold. e.g., 4 cores, then [P_1core, P_2cores, P_3cores, P_4cores]
        self.trip_freq                  = []    # List of frequencies for the thermal trip points
        self.DTPM_trip_freq             = []    # List of frequencies for the thermal trip points (custom DTPM policies)
        self.OPP                        = []    # List of all Operating Performance Points (OPPs), each OPP is a <frequency, voltage> tuple.
        self.acc_config_list            = []    # List of all supported configurations (only applicable to accelerators)
        self.acc_current_config         = ''    # Indicate the current configuration (only applicable to accelerators)
        self.current_frequency          = 0     # Indicate the current cluster frequency (may be affected by throttling)
        self.policy_frequency           = 0     # Indicate the frequency defined by the defined policy
        self.current_voltage            = 0     # Indicate the current cluster voltage
        self.num_active_cores           = 0     # Indicate the number of active cores
        self.num_total_cores            = 0     # Indicate the number of total cores
        self.current_power_cluster      = 0     # Indicate the current power dissipation for the cluster (dynamic + static)
        self.snippet_power_list         = []    # List of power values for the current snippet
        self.snippet_num_tasks_list     = []    # List of num_tasks for the current snippet
        self.current_acc_kernels_exec   = []    # List of kernels that are being executed in the accelerator - Only for tasks that are currently running
        self.current_acc_configuration  = []    # List of current acc configurations - For DAP, it includes both currently running and ended tasks (if the DAP was not reconfigured yet) (DAP only)
        self.DAP_utilization_config     = 0     # Current DAP utilization for the current configuration (# sub-PEs) (DAP only)
        self.DAP_utilization_exec       = 0     # Current DAP utilization for the current kernels being executed (# sub-PEs) (DAP only)
        self.reconfiguration_overhead   = 0     # Current DAP reconfiguration overhead (DAP only)
