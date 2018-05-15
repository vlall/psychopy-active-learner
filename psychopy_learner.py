import random
import math
import pandas as pd
from psychopy import visual, event, core
import dot_experiment
from bams.learners import ActiveLearner
from bams.query_strategies import (
    BALD,
    HyperCubePool,
    RandomStrategy,
)
import numpy as np


# Set active learner parameters
NDIM = 1
POOL_SIZE = 500
BUDGET = 100
BASE_KERNELS = ["PER", "LIN"]
DEPTH = 2
win = visual.Window(
size=[500, 500],
units="pix",
fullscr=False
)

def scale_up(threshold, dim):
    """Rescale up to actual values"""
    out = int(dim * threshold)
    if out == 0:
        return 1
    return out


def scale_down(threshold, dim):
    """Rescale 0 <= output <= 1"""
    out = float(dim/threshold) if threshold else 0.0
    return out


def oracle(x):
    """Run a psychopy experiment by scaling up the features so they can be used as input.
    Then scale down the output for the active learner."""

    max_n_dots = 100
    # Scale up
    n_dots = scale_up(max_n_dots, float(x[0]))
    # No need to scale contrast
    contrast = .9 #float(x[1])
    answer_text = visual.TextStim(win)
    guess = dot_experiment.run_experiment(win, answer_text, n_dots, contrast)
    #score = 1 - (abs(float(guess)-float(n_dots)) / float(n_dots))
    return (n_dots/100.0)


def random_oracle(x):
    x = random.random()
    print(x)
    return x

if __name__ == "__main__":
    learner = ActiveLearner(
        query_strategy=BALD(pool=HyperCubePool(NDIM, POOL_SIZE)),
        budget=BUDGET,
        base_kernels=BASE_KERNELS,
        max_depth=DEPTH,
        ndim=NDIM,
    )
    trial = 0
    threshold = 0.01
    outterDict = {}
    posteriorMatrix = np.zeros((BUDGET, len(learner.models)))
    maxModel = []
    while learner.budget > 0:
        x = learner.next_query()
        y = learner.query(oracle, x)
        learner.update(x, y)
        print(trial)
        posteriors = learner.posteriors
        for i, model in enumerate(learner.models):
            posteriorMatrix[trial, i] = posteriors[i]
        trial+=1
    df = pd.DataFrame(posteriorMatrix, columns=[str(i) for i in learner.models])
    print(df)
    df.to_pickle("BALD_1.pkl")