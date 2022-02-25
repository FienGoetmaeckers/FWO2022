# FWO2022
Python interactive code to explore the paradigm and the computational model discussed in Fien Goetmaeckers' FWO PhD FR project proposal.

Execute Experiment_runner.py. Modules create_grids.py and solving_models.py are needed to run the program. Codes are operable in Python 3, needed modules are:

* numpy (https://numpy.org/install/)
* matplotlib (https://matplotlib.org/stable/users/installing/index.html)
* seaborn (https://seaborn.pydata.org/installing.html) 
* scikit (https://scikit-learn.org/stable/install.html)

When excecuting Experiment_runner.py, you can either
1) Solve the experiment yourself and try to accumulate the highest reward
2) See an agent sample randomly
3) See an agent execute the GPR-UCB model to generalize the observed rewards and optimistically balance exploration with exploitation.
