import random
import pandas as pd
from psychopy import visual, event, core
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
import unittest


class TestStringMethods(unittest.TestCase):


    mapping = "../mappings/mapping_example.json"
    configFile = "../config.txt"


    def test_read_config():
        # Read the config file
        parser = configparser.ConfigParser()
        parser.read(configFile)
        configData = {}
        for section in parser.sections():
            configData.update(dict(parser.items(section)))
        print(configData)


    def test_set_config_values():
        # Set the config values
        finalData = pd.DataFrame(columns=['n_dots', 'contrast', 'guess', 'n_dim'])
        POOL_SIZE = 200
        BUDGET = 1
        BASE_KERNELS = [E]
        DEPTH = 1
        DATA_PATH = configData["data_path"]


    def test_outputfiles():
        random_df = pd.read_pickle("%s.pkl" % RANDOM_PATH)
        assert(len(random_df)>0)


    def test_config_parser():
        pass


    def test_dummy_oracle():
        dummy = dummy_oracle(0.49)
        self.assertEqual(scaled, 0.49)

    def test_scale_up():
        scaled = scale_up(100, 0.99)
        self.assertEqual(scaled, 99)

    def test_scale_down():
        scaled = scale_down(100, 99)
        self.assertEqual(scaled, 0.99)


if __name__ == '__main__':
    unittest.main()
