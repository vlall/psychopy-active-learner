import random
import math
from psychopy import visual, event, core
import dot_experiment
from bams.learners import ActiveLearner
from bams.query_strategies import (
    # BALD,
    HyperCubePool,
    RandomStrategy,
)

# Set active learner parameters
NDIM = 3
POOL_SIZE = 500
BUDGET = 100
BASE_KERNELS = ["PER", "LIN"]
DEPTH = 2

# Set psychopy variables
win = visual.Window(
size=[500, 500],
units="pix",
fullscr=False
)

def scale_up(threshold, dim):
    """Rescale up to actual values"""
    out = int(dim * threshold)
    return out

def scale_down(threshold, dim):
    """Rescale 0 <= output <= 1"""
    out = float(dim/threshold) if threshold else 0.0
    return out

def oracle(x):
    """Run a psychopy experiment by scaling up the features so they can be used as input.
    Then scale down the output for the active learner.
    """
    max_n_dots = 100
    # Scale up
    n_dots = scale_up(max_n_dots, float(x[0]))
    # No need to scale contrast
    contrast = float(x[1])
    answer_text = visual.TextStim(win)
    guess = dot_experiment.run_experiment(win, answer_text, n_dots, contrast)
    #score = 1 - (abs(float(guess)-float(n_dots)) / float(n_dots))
    return (n_dots/100.0)

learner = ActiveLearner(
    query_strategy=RandomStrategy(pool=HyperCubePool(NDIM, POOL_SIZE)),
    budget=BUDGET,
    base_kernels=BASE_KERNELS,
    max_depth=DEPTH,
    ndim=NDIM,
)

trials = 0
threshold = 0.02
while learner.budget > 0:
    trials+=1
    x = learner.next_query()
    y = learner.query(oracle, x)
    learner.update(x, y)
    if trials > 9:
        posteriors = learner.posteriors
        for i, model in enumerate(learner.models):
            print(model)
            if posteriors[i] > threshold:
                print(posteriors[i])
                print("")
        print("*****")
