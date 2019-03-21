# Psychopy Active Learner

This project runs a basic numerosity experiment using the Psychopy Toolbox, which presents
a uniform distribution of 0<=x<=100 dots across a 500x500 window. The `dot_experiment.py` file takes a input the `n_dots` being displayed and the `contrast` of those dots. This is used by the Bayesian
active model selection code in `psychopy_learner.py` which manipulates a combination of these variables
in order to converge to the kernel grammar with the highest confidence. This grammar is used to describe the relationship between stimuli and human behavior.

After the experiment is run to completion using the configured hyper-parameters, the experiment generates a `data/` folder along with a UUID used to denote the specific experiment data generated. This is used as input into the analysis scripts for interpretting the data.
 
