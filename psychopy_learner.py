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


class PsychopyLearner(object):
    """
    Attributes are set from the config.txt

    Args:
        None

    Attributes:
        STRATEGIES (list[str]): The active learner objects from BAMS which we use in the experiment
        MANIPULATIONS (dict): The variables the learner is manipulating. The key is the name of the
            manipulated variable. The value is the number of dimensions being manipulated.
        POOL_SIZE (int): Pool size tells you the maximum amount of candidate models
            (TODO: Check if this is correct! Pool size might be doing something different*)
        BUDGET (int): The budget sets the amount of iterations.
        BASE_KERNELS (list[str]): Exception error code.
        DEPTH (int): Exception error code.
        DATA_PATH (str): The data output path.
        HUMAN (bool): True sets this to the regular `PsychopyLearner.oracle()`,
            False sets this to the `Psychopy.dummer_oracle()`.
    """

    def __init__(self):
        # Read the config file
        configFile = "config.txt"
        parser = configparser.ConfigParser()
        parser.read(configFile)
        configData = {}
        for section in parser.sections():
            configData.update(dict(parser.items(section)))

        # Set the config values
        self.STRATEGIES = list(configData["strategies"].split(", "))
        self.MANIPULATIONS = json.loads((configData["manipulations"]))
        self.POOL_SIZE = int(configData["pool_size"])
        self.BUDGET = int(configData["budget"])
        self.BASE_KERNELS = list(configData["base_kernels"].split(", "))
        self.DEPTH = int(configData["depth"])
        self.DATA_PATH = configData["data_path"]
        self.HUMAN = str(configData["human"])

        # Initialize data structures
        self.json_uuid = {}
        self.finalData = pd.DataFrame(columns=['n_dots', 'contrast', 'guess', 'n_dim'])

        # Check if we need to load a psychopy window
        if self.HUMAN == 'True':
            win = visual.Window(
                size=[500, 500],
                units="pix",
                fullscr=False
            )
        else:
            self.oracle = self.dummy_oracle

    def run_matrix(self):
        """
        Experiment matrix (manipulations x strategies). See config.txt for these settings
        """
        for self.manipulation, dim in self.MANIPULATIONS.items():
            for strategy in self.STRATEGIES:
                val = self.run_learner_on_experiment(strategy, dim, self.manipulation)
                self.json_uuid['%s_%s' % (strategy, self.manipulation)] = val
                # Clear data for next strategy
                self.finalData = self.finalData[0:0]

        # Generate and dump mappings of the experimental data.
        mapId = 'mapping_%s.json' % str(uuid.uuid4())[:4]
        if not os.path.exists("mappings"):
            os.makedirs("mappings")
        mapPath = "mappings/%s" % (mapId)
        with open(mapPath, 'w') as map:
            json.dump(self.json_uuid, map)
        print("Your metadata was saved to mappings/%s " % (mapId))
        return mapPath

    def scale_up(self, threshold, dim):
        """Rescale up to actual values"""
        out = round(dim * threshold)
        if out == 0:
            return 1
        return int(out)

    def scale_down(self, threshold, dim):
        """Rescale 0 <= output <= 1"""
        out = float(dim / float(threshold)) if threshold else 0.0
        return out

    def oracle(self, x):
        """
        Run a psychopy experiment by scaling up the features so they can be used as input.
        Then scale down the output for the active learner.
        """
        max_n_dots = 100
        # Scale up
        if self.manipulation == 'dots' or self.manipulation == 'all':
            n_dots = self.scale_up(max_n_dots, x[0])
        else:
            n_dots = self.scale_up(max_n_dots, random.random())
        # No need to scale contrast
        if self.manipulation == 'contrast':
            contrast = x[0]
        elif self.manipulation == 'all':
            contrast = x[1]
        else:
            contrast = random.random()
        answer_text = visual.TextStim(self.win)
        guess = dot_experiment.run_experiment(self.win, answer_text, n_dots, contrast)
        self.finalData.loc[len(self.finalData)] = [n_dots, contrast, int(guess), list(x)]
        print(self.finalData)
        if guess:
            return float(guess) / 100.0
        return 0.0

    def dummy_oracle(self, x):
        """
        The oracle usually manipulates the dimension(s) in x based on prior information.
        This dummy oracle does not manipulate any dimensions, it simply returns the dots displayed
        """
        max_n_dots = 100
        # Scale up
        if self.manipulation == 'dots' or self.manipulation == 'all':
            n_dots = self.scale_up(max_n_dots, x[0])
        else:
            n_dots = self.scale_up(max_n_dots, random.random())
        if self.manipulation == 'contrast':
            contrast = x[0]
        elif self.manipulation == 'all':
            contrast = x[1]
        else:
            contrast = random.random()
        self.finalData.loc[len(self.finalData)] = [n_dots, contrast, n_dots, list(x)]
        print(self.finalData)
        return self.scale_down(max_n_dots, n_dots)

    def translate(self, model):
        s = str(model).split()[0]
        translate = s.replace("(", "_").rstrip(',')
        return translate

    def run_learner_on_experiment(self, strategy, dim, manipulation):
        UUID = str(uuid.uuid4())
        PATH = self.DATA_PATH + UUID + "/"
        NAME = "%s_%s" % (strategy, manipulation)
        if strategy.lower() == "bald":
            print("Running %s on %s" % (strategy, str(manipulation)))
            learner = ActiveLearner(
                query_strategy=BALD(pool=HyperCubePool(dim, self.POOL_SIZE)),
                budget=self.BUDGET,
                base_kernels=self.BASE_KERNELS,
                max_depth=self.DEPTH,
                ndim=dim,
            )

        elif strategy.lower() == "random":
            print("Running %s on %s" % (strategy, str(manipulation)))
            learner = ActiveLearner(
                query_strategy=RandomStrategy(pool=HyperCubePool(dim, self.POOL_SIZE)),
                budget=self.BUDGET,
                base_kernels=self.BASE_KERNELS,
                max_depth=self.DEPTH,
                ndim=dim,
            )
        else:
            print("%s is not a valid strategy (choose either BALD or Random). Exiting." % strategy)
            sys.exit()
        trial = 0
        posteriorMatrix = np.zeros((self.BUDGET, len(learner.models)))
        while learner.budget > 0:
            x = learner.next_query()
            y = learner.query(self.oracle, x)
            learner.update(x, y)
            posteriors = learner.posteriors
            for i, model in enumerate(learner.models):
                posteriorMatrix[trial, i] = posteriors[i]
            trial += 1
        for model in learner.models:
            translate = self.translate(str(model))
            make_folder = "%s/all_models/%s" % (PATH, NAME)
            if not os.path.exists(make_folder):
                os.makedirs(make_folder)
            filepath = "%s/%s.pkl" % (make_folder, translate)
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print("posterior saved as:")
            [print(str(i)) for i in learner.models]
        print("string saved as:")
        [print(str(i)) for i in learner.models]
        df = pd.DataFrame(posteriorMatrix, columns=[str(i) for i in learner.models])
        outputName = PATH + NAME
        df.to_pickle(outputName + '.pkl')
        self.finalData.to_pickle(outputName + '_trials.pkl')
        runtime_data = {
            'NAME': NAME,
            "DATE/TIME": strftime("%Y-%m-%d %H:%M:%S", gmtime()),
            "PATH": PATH,
            "NAME": NAME,
            "UUID": UUID,
            "NDIM": dim,
            "MANIPULATE": manipulation,
            "POOL_SIZE": self.POOL_SIZE,
            "BUDGET": self.BUDGET,
            "BASE_KERNELS": self.BASE_KERNELS,
            "DEPTH": self.DEPTH,
            "OUTPUT_NAME": outputName,
        }
        with open(outputName + '_runtime.json', 'w') as outfile:
            json.dump(runtime_data, outfile)
        print("****** Finished. Use the following path as input for the plot: ******")
        print(outputName)
        return (UUID)


if __name__ == "__main__":
    from psychopy import visual, event, core
    import dot_experiment
    experiments = PsychopyLearner()
    experiments.run_matrix()
