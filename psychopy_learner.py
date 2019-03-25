import random
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
import pickle
import uuid
import os
import json
import sys
from time import gmtime, strftime
from six.moves import configparser


# Read the config file
configFile = "config.txt"
parser = configparser.ConfigParser()
parser.read(configFile)
configData = {}
for section in parser.sections():
    configData.update(dict(parser.items(section)))

# Set the config values
finalData = pd.DataFrame(columns=['n_dots', 'contrast', 'guess', 'n_dim'])
NAME = configData["name"]
MANIPULATE = configData["manipulate"]
NDIM = int(configData["ndim"])
POOL_SIZE = int(configData["pool_size"])
BUDGET = int(configData["budget"])
BASE_KERNELS = list(configData["base_kernels"].split(", "))
DEPTH = int(configData["depth"])
DATA_PATH = configData["data_path"]
UUID = str(uuid.uuid4())
PATH = DATA_PATH + UUID + "/"
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
    if MANIPULATE=='dots' or MANIPULATE=='all':
        n_dots = scale_up(max_n_dots, x[0])
    else:
        n_dots = scale_up(max_n_dots, random.random())
    # No need to scale contrast
    if MANIPULATE=='contrast':
        contrast = x[0]
    elif MANIPULATE=='all':
        contrast = x[1]
    else:
        contrast = random.random()
    answer_text = visual.TextStim(win)
    guess = dot_experiment.run_experiment(win, answer_text, n_dots, contrast)
    finalData.loc[len(finalData)] = [n_dots, contrast, int(guess), list(x)]
    print(finalData)
    if guess:
        return float(guess)/100.0
    return 0.0


def dummy_oracle(x):
    """Doesn't manipulate data, just records as if it were a real trial."""
    max_n_dots = 100
    # Scale up
    if MANIPULATE=='dots' or MANIPULATE=='all':
        n_dots = scale_up(max_n_dots, x[0])
    else:
        n_dots = scale_up(max_n_dots, random.random())
    if MANIPULATE=='contrast':
        contrast = x[0]
    elif MANIPULATE=='all':
        contrast = x[1]
    else:
        contrast = random.random()
    finalData.loc[len(finalData)] = [n_dots, contrast, n_dots, list(x)]
    print(finalData)
    return x[0]


def run_learner_on_experiment(strategy):
    strategy_name = NAME.split("_")[0]
    if strategy_name.lower() == "bald":
        print("Running %s %s" % (strategy_name, str(NDIM)))
        learner = ActiveLearner(
            query_strategy=BALD(pool=HyperCubePool(NDIM, POOL_SIZE)),
            budget=BUDGET,
            base_kernels=BASE_KERNELS,
            max_depth=DEPTH,
            ndim=NDIM,
        )

    elif strategy_name.lower() == "random":
        print("Running %s %s" % (strategy_name, str(NDIM)))
        learner = ActiveLearner(
            query_strategy=RandomStrategy(pool=HyperCubePool(NDIM, POOL_SIZE)),
            budget=BUDGET,
            base_kernels=BASE_KERNELS,
            max_depth=DEPTH,
            ndim=NDIM,
        )
    else:
        print("%s is not a valid strategy (choose either BALD or Random). Exiting." % strategy_name)
        sys.exit()

    trial = 0
    threshold = 0.01
    outterDict = {}
    posteriorMatrix = np.zeros((BUDGET, len(learner.models)))
    maxModel = []
    while learner.budget > 0:
        x = learner.next_query()
        y = learner.query(dummy_oracle, x)
        learner.update(x, y)
        print(trial)
        posteriors = learner.posteriors
        for i, model in enumerate(learner.models):
            posteriorMatrix[trial, i] = posteriors[i]
            if learner.budget==1:
                s = str(model).split()[0]
                translate = s.replace("(", "_").rstrip(',')
                make_folder = "%s/all_models/%s" % (PATH, NAME)
                if not os.path.exists(make_folder):
                    os.makedirs(make_folder)
                filepath = "%s/%s.pkl" % (make_folder, translate)
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
        trial+=1
    df = pd.DataFrame(posteriorMatrix, columns=[str(i) for i in learner.models])
    outputName = PATH + NAME
    df.to_pickle(outputName + '.pkl')
    finalData.to_pickle(outputName + '_trials.pkl')
    runtime_data = {
        'NAME': NAME,
        "DATE/TIME": strftime("%Y-%m-%d %H:%M:%S", gmtime()),
        "PATH": PATH,
        "NAME": NAME,
        "UUID": UUID,
        "NDIM": NDIM,
        "MANIPULATE": MANIPULATE,
        "POOL_SIZE": POOL_SIZE,
        "BUDGET": BUDGET,
        "BASE_KERNELS": BASE_KERNELS,
        "DEPTH": DEPTH,
        "OUTPUT_NAME": outputName,
    }
    with open(outputName + '_runtime.json', 'w') as outfile:
        json.dump(runtime_data, outfile)
    print("****** Finished. Use the following path as input for the plot: ******")
    print(outputName)
    return(UUID)

json_uuid = {}
strategies = ["BALD_1", "Random_1"]
for strategy in strategies:
    uuid = run_learner_on_experiment(strategy)
    json_uuid[strategy] = uuid
print(json_uuid)