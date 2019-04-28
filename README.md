# Psychopy Active Learner

[![Build Status](https://travis-ci.com/vlall/psychopy-active-learner.svg?token=u4sdN1vvyVBZq3MUz13n&branch=master)](https://travis-ci.com/vlall/psychopy-active-learner)

This project runs a basic numerosity experiment which presents
a uniform distribution of 0<=x<=100 dots across a 500x500 window. Hyper-parameters are set in the `config.txt`. During the normal `oracle` execution, `dot_experiment.py` file takes as input the `n_dots` being pressented and the `contrast` of those dots. During each trial of presentation, the Bayesian
active model selection code in `psychopy_learner.py` manipulates certain varaibles using the experiment in `dot_experiment.py` to learn the relationship between the manipulation present and the dots guessed. This relationship is modelled using convergence to a certain *kernel grammar* within a *budgeted* trials. These variables are set in the `config.txt`. This kernal grammar is used to describe the relationship between stimuli and human behavior.

After the experiment is run to completion using the configured hyper-parameters, the experiment generates a `data/` folder along with a UUID used to denote the specific experiment data generated. This is used as input into the analysis scripts for interpretting the data. The program generates a metadata mapping to all of the experiments run in `mappings/`
 

### Prerequisites

Make sure you have the BAMS package and it's requirements installed (see: https://github.com/Dallinger/bams). Then, you can install the `psychopy-active-learner` requirements using `pip install -r requirements.txt` from this cloned repo.


### Running

Check to make sure you have the desired configuration in `config.txt`.
This the example config:

```
# These are active learner objects from BAMS. Each strategy is run against the manipulations below
STRATEGIES = BALD, Random

# The manipuations refer to a dictionary of all of the variables the learner is manipulating. The key is the name of the manipulated variable. The value is the number of dimensions being simulatenously manipulated.
MANIPULATIONS = {"dots": 1, "contrast": 1, "random": 1, "all": 3}

# True sets this to the regular `PsychopyLearner.oracle()`, False sets this to the `Psychopy.dummer_oracle()`
HUMAN = False

# Pool size tells you granularity of learner outputs for estimating the function using Sobol generation.
POOL_SIZE = 200

# The budget sets the amount of iterations
BUDGET = 50

# The base kernels descrive the kernels to use, later combined
BASE_KERNELS = PER, LIN, K, LG

# The depth specifies the degree to which the base kerneles are combined
DEPTH = 2

# This is the data output path
DATA_PATH = data/

```
Once everything is setup you can run the experiment matrix using `python psychopy_learner.py`


### Running the tests

Use `python -m pytest` to run from the `tests` directory

### References:
Gardner, J., Malkomes, G., Garnett, R., Weinberger, K. Q., Barbour, D., & Cunningham, J. P. (2015). Bayesian active model selection with an application to automated audiometry. In Advances in Neural Information Processing Systems (pp. 2386-2394).

Malkomes, G., Schaff, C., & Garnett, R. (2016). Bayesian optimization for automated model selection. In Advances in Neural Information Processing Systems (pp. 2900-2908).
