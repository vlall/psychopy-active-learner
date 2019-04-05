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
import unittest


class TestStringMethods(unittest.TestCase):


    mapping = "../mappings/mapping_example.json"
    config = "../config.txt"


    def test_outputfiles:
        PATH =
        random_df = pd.read_pickle("%s.pkl" % RANDOM_PATH)
        assert(len(random_df)>0)


    def test_config_parser:
        pass

    def test_oracle():
        pass


    def test_dummy_oracle():
        pass


    def test_scale_up():
        pass


    def test_scale_down():
        pass


    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')


    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())


    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()
