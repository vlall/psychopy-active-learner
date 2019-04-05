import random
import pandas as pd
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
from psychopy import visual, event, core
import dot_experiment


def scale_up(threshold, dim):
    """Rescale up to actual values"""
    out = round(dim * threshold)
    if out == 0:
        return 1
    return int(out)


def scale_down(threshold, dim):
    """Rescale 0 <= output <= 1"""
    out = float(dim / float(threshold)) if threshold else 0.0
    return out


def oracle(x):
    """
    Run a psychopy experiment by scaling up the features so they can be used as input.
    Then scale down the output for the active learner.
    """
    max_n_dots = 100
    # Scale up
    if manipulation == 'dots' or manipulation == 'all':
        n_dots = scale_up(max_n_dots, x[0])
    else:
        n_dots = scale_up(max_n_dots, random.random())
    # No need to scale contrast
    if manipulation == 'contrast':
        contrast = x[0]
    elif manipulation == 'all':
        contrast = x[1]
    else:
        contrast = random.random()
    answer_text = visual.TextStim(win)
    guess = dot_experiment.run_experiment(win, answer_text, n_dots, contrast)
    finalData.loc[len(finalData)] = [n_dots, contrast, int(guess), list(x)]
    print(finalData)
    if guess:
        return float(guess) / 100.0
    return 0.0


def dummy_oracle(x):
    """
    The oracle usually manipulates the dimension(s) in x based on prior information.
    This dummy oracle does not manipulate any dimensions, it simply returns the dots displayed
    """
    max_n_dots = 100
    # Scale up
    if manipulation == 'dots' or manipulation == 'all':
        n_dots = scale_up(max_n_dots, x[0])
    else:
        n_dots = scale_up(max_n_dots, random.random())
    if manipulation == 'contrast':
        contrast = x[0]
    elif manipulation == 'all':
        contrast = x[1]
    else:
        contrast = random.random()
    finalData.loc[len(finalData)] = [n_dots, contrast, n_dots, list(x)]
    print(finalData)
    return scale_down(max_n_dots, n_dots)


def run_learner_on_experiment(strategy, dim, manipulation):
    UUID = str(uuid.uuid4())
    PATH = DATA_PATH + UUID + "/"
    NAME = "%s_%s" % (strategy, manipulation)
    if strategy.lower() == "bald":
        print("Running %s on %s" % (strategy, str(manipulation)))
        learner = ActiveLearner(
            query_strategy=BALD(pool=HyperCubePool(dim, POOL_SIZE)),
            budget=BUDGET,
            base_kernels=BASE_KERNELS,
            max_depth=DEPTH,
            ndim=dim,
        )

    elif strategy.lower() == "random":
        print("Running %s on %s" % (strategy, str(manipulation)))
        learner = ActiveLearner(
            query_strategy=RandomStrategy(pool=HyperCubePool(dim, POOL_SIZE)),
            budget=BUDGET,
            base_kernels=BASE_KERNELS,
            max_depth=DEPTH,
            ndim=dim,
        )
    else:
        print("%s is not a valid strategy (choose either BALD or Random). Exiting." % strategy)
        sys.exit()
    trial = 0
    posteriorMatrix = np.zeros((BUDGET, len(learner.models)))
    while learner.budget > 0:
        x = learner.next_query()
        y = learner.query(oracle, x)
        learner.update(x, y)
        print(trial)
        posteriors = learner.posteriors
        for i, model in enumerate(learner.models):
            posteriorMatrix[trial, i] = posteriors[i]
            if learner.budget == 1:
                s = str(model).split()[0]
                translate = s.replace("(", "_").rstrip(',')
                make_folder = "%s/all_models/%s" % (PATH, NAME)
                if not os.path.exists(make_folder):
                    os.makedirs(make_folder)
                filepath = "%s/%s.pkl" % (make_folder, translate)
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
        trial += 1
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
        "NDIM": dim,
        "MANIPULATE": manipulation,
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
    return (UUID)


# Read the config file
configFile = "config.txt"
parser = configparser.ConfigParser()
parser.read(configFile)
configData = {}
for section in parser.sections():
    configData.update(dict(parser.items(section)))

# Set the config values
POOL_SIZE = int(configData["pool_size"])
BUDGET = int(configData["budget"])
BASE_KERNELS = list(configData["base_kernels"].split(", "))
DEPTH = int(configData["depth"])
DATA_PATH = configData["data_path"]
STRATEGIES = list(configData["strategies"].split(", "))
MANIPULATIONS = json.loads((configData["manipulations"]))
HUMAN = str(configData["human"])

# Initialize data structures
json_uuid = {}
finalData = pd.DataFrame(columns=['n_dots', 'contrast', 'guess', 'n_dim'])

# Check if we need to load a psychopy window
if HUMAN == 'True':
    win = visual.Window(
        size=[500, 500],
        units="pix",
        fullscr=False
    )
else:
    oracle = dummy_oracle

# Experiment matrix (manipulations x strategies). See config.txt for these settings
for manipulation, dim in MANIPULATIONS.items():
    for strategy in STRATEGIES:
        val = run_learner_on_experiment(strategy, dim, manipulation)
        json_uuid['%s_%s' % (strategy, manipulation)] = val
        # Clear data for next strategy
        finalData = finalData[0:0]

# Generate and dump a mapping to the experiment's data.
mapId = 'mapping_%s.json' % str(uuid.uuid4())[:4]
if not os.path.exists("mappings"):
    os.makedirs("mappings")
mapPath = "mappings/%s" % (mapId)
with open(mapPath, 'w') as map:
    json.dump(json_uuid, map)
print("Your metadata was saved to mappings/%s " % (mapId))
