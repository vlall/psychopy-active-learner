# Psychopy Active Learner

[![Build Status](https://travis-ci.com/vlall/psychopy-active-learner.svg?token=u4sdN1vvyVBZq3MUz13n&branch=master)](https://travis-ci.com/vlall/psychopy-active-learner)

This project uses the [BAMS library](https://www.github.com/dallinger/bams) to generate automated end-to-end human bahavioral experiments that use active learning to model, predict, and graph human behavior.

As an example, we run a basic numerosity experiment which presents a uniform distribution of `1-100` dots across a `500x500` window. During the normal execution, the `dot_experiment.py` file takes as input the `n_dots` being presented and the `contrast` of those dots. During each trial, the Bayesian active model selection code in `psychopy_learner.py` manipulates certain varaibles using the experiment in `dot_experiment.py`. These manipulations are then exposed to the active learning strategy, which then tries to learn the relationship between the manipulation and human behavior. Finally, this relationship is modeled using the posterior probability's convergence to a certain *kernel grammar* within a pre-defined *budget* of trials. All hyper-parameters are set in the `config.txt`, which define the search space and strategies used by the learner. The model's convergence to a kernel grammar is used to describe the relationship between stimuli and human behavior. This object can then be queried for predictions using unseen data points.

After the experiment matrix is run to completion using the configured hyper-parameters, the experiment generates a folder along with a `UUID` used to denote the specific experiment data generated. This is used as input into the analysis scripts for interpretting the data. The total matrix generates a metadata mapping to all of the individual experiments run, and saves this reference to all experiments in the matrix to `mappings/`
 

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
