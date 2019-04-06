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
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psychopy_learner import PsychopyLearner


class TestPsychopyLearner(unittest.TestCase):


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


    def test_bald_path(self):
        root = "data/"
        BALD_PATH_ROOT = root + "21605e4d-258b-4acf-8f82-7a0bc62f83ed/BALD_dots"
        BALD_PATH_ALL = BALD_PATH_ROOT + "/all_models"
        random_df = pd.read_pickle("%s.pkl" % BALD_PATH_ROOT)
        assert(len(random_df)>0)


    def test_random_path(self):
        root = "data/"
        RANDOM_PATH_ROOT = root + "fec5ab43-9fac-4943-b8a7-2ef575aced28/Random_dots"
        RANDOM_PATH_ALL = RANDOM_PATH_ROOT + "/all_models"
        random_df = pd.read_pickle("%s.pkl" % RANDOM_PATH_ROOT)
        assert(len(random_df)>0)


    def test_scale_up(self):
        experiments = PsychopyLearner()
        scaled = experiments.scale_up(100, 0.99)
        self.assertEqual(scaled, 99)


    def test_scale_down(self):
        experiments = PsychopyLearner()
        scaled = experiments.scale_down(100, 99)
        self.assertEqual(scaled, 0.99)


    """
    def test_run_learner_on_experiment(self):
        mapping_output = run_learner_on_experiment('BALD', 1, "dots")
        assert os.path.isfile("mappings/%s" % mapping_output)

    def test_dummy_oracle(self):
        dummy = dummy_oracle(0.49)
        self.assertEqual(scaled, 0.49)
    """

if __name__ == '__main__':
    unittest.main()
