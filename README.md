# Psychopy Active Learner

[![Build Status](https://travis-ci.com/vlall/psychopy-active-learner.svg?token=u4sdN1vvyVBZq3MUz13n&branch=master)](https://travis-ci.com/vlall/psychopy-active-learner)

This project runs a basic numerosity experiment using the Psychopy Toolbox, which presents
a uniform distribution of 0<=x<=100 dots across a 500x500 window. Hyper parameters are set in the `config.txt`. The `dot_experiment.py` file takes as input the `n_dots` being pressented and the `contrast` of those dots. During each trial of presentation, the Bayesian
active model selection code in `psychopy_learner.py` manipulates the experiment to best converge to a certain kernel grammar within a budgeted trial limit.
This grammar is used to describe the relationship between stimuli and human behavior.

After the experiment is run to completion using the configured hyper-parameters, the experiment generates a `data/` folder along with a UUID used to denote the specific experiment data generated. This is used as input into the analysis scripts for interpretting the data. The program generates a metadata mapping to all of the experiments run in `mappings/`
 
To run, make sure you have BAMS installed (https://github.com/Dallinger/bams)

Run:

`pip install -r requirements.txt`

Make sure `config.txt` has the desired hyper-parameters, then run the experiment using
`python psychopy_learner.py`