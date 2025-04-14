@mainpage Overview

************************************************************************
    © Copyright 2020 ASU All rights reserved. \n
    This file contains confidential and proprietary information of DASH-SoC Project.
************************************************************************

**DASH-Sim** is a system-level domain-specific system-on-chip simulation framework. \n
The goal of **DASH-Sim** is to enable:
    - Run-time scheduling algorithm development, 
    - %DTPM policy design, and 
    - Rapid design space exploration.
    
To achieve these goals, it provides:
    - *Scalability*: Provide the ability to simulate instances of multiple applications simultaneously by streaming multiple jobs from a pool of active domain applications.
    - *Flexibility*: Enable the end-users to specify the SoC configuration, target applications, and the resource database swiftly (e.g., in minutes) using simple interfaces.
    - *Modularity*: Enable algorithm developers to modify the existing scheduling and %DTPM algorithms, and add new algorithms with minimal effort.
    - *User-friendly Productivity Tools*: Provide built-in capabilities to collect, report and plot key statistics, including power dissipation, execution time, throughput, energy consumption, and temperature.
    

@image html DS3_framework.png Organization of DASH-Sim framework describing the inputs and key functional components. height=30% width=30% 