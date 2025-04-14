@page install Installation
@tableofcontents

@section windows Installing on Windows
    - <a href=https://www.anaconda.com/products/individual>Download</a> **Anaconda** installer and install it to your machine
        -# To uninstall an older version of Anaconda in your machine, if needed, see the instructions on this <a href=https://docs.anaconda.com/anaconda/install/uninstall/>link</a>
    - Open Anaconda prompt (by typing in SEARCH BOX) and
        -# Install tensorflow by following the instructions on this <a href=https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/>link</a>
            - Then in the **tensorflow environment** install following packages with *pip*:
                - \code{.py} pip install simpy pandas numpy networkx scipy matplotlib ipython pytest scikit_learn \endcode
            - Then install following pacakges with *conda*:
                - Install pydot with \code{.py} conda install -c anaconda pydot \endcode
                - Install cplex and docplex with \code{.py} conda install -c ibmdecisionoptimization docplex cplex \endcode
                - Install keras with \code{.py} conda install -c anaconda keras \endcode
        
        
@warning The installed CPLEX engine is a **Community Edition** with limited solving capabilities. If you are within limits, then any CPLEX based code will run perfectly, otherwise it will fail with errors. \n 
Note that some of the docplex use cases might be above these limits, such as dynamic scheduling for streaming applications.







