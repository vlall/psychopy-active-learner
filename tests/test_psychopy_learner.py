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
from six.moves import configparser
import unittest


class TestPsychopyLearner(object):
    """TODO: Import psychopy_learner.py without errors"""

    # strategies = ["BALD", "Random"]
    # manipulations = {"dots": 1, "contrast": 1, "random" :1 , "all": 3}
    # dim = 1
    # manipulate = "dots"
    # root = "data/"
    # strategies = ["BALD_%s" % manipulate, "Random_%s" % manipulate]
    # BALD_PATH_ROOT = root + mapping_example["BALD_" + manipulate]
    # BALD_PATH_ALL = BALD_PATH_ROOT + "/all_models"
    # RANDOM_PATH_ROOT = root + mapping_example["Random_" + manipulate]
    # RANDOM_PATH_ALL = RANDOM_PATH_ROOT + "/all_models"
    # trialData = ['n_dots', 'contrast', 'guess', 'n_dim']


    def test_config_keys(self):
        configFile = "config.txt"
        mapId = "mappings/mapping_example.json"
        with open(mapId) as json_file:
            mapping_example = json.load(json_file)

        parser = configparser.ConfigParser()
        parser.read(configFile)
        configData = {}
        for section in parser.sections():
            configData.update(dict(parser.items(section)))
        keys = ['strategies', 'manipulations', 'human', 'pool_size', 
                'budget', 'base_kernels', 'depth', 'data_path'
        ]
        assert set(keys).issubset(list(configData))


    def test_run_learner_on_experiment(self):
        mapping_output = run_learner_on_experiment('BALD', 1, "dots")
        assert os.path.isfile("mappings/%s" % mapping_output)


    def test_bald_path(self):
        random_df = pd.read_pickle("%s.pkl" % BALD_PATH_ROOT)
        assert(len(random_df)>0)


    def test_random_path(self):
        random_df = pd.read_pickle("%s.pkl" % RANDOM_PATH_ROOT)
        assert(len(random_df)>0)


    def test_dummy_oracle(self):
        dummy = dummy_oracle(0.49)
        self.assertEqual(scaled, 0.49)


    def test_scale_up(self):
        scaled = scale_up(100, 0.99)
        self.assertEqual(scaled, 99)


    def test_scale_down(self):
        scaled = scale_down(100, 99)
        self.assertEqual(scaled, 0.99)


if __name__ == '__main__':
    unittest.main()
