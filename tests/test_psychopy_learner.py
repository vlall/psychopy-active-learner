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


def test_outputfiles:
    RANDOM_PATH = "data/066eb6db-0440-4cdd-8664-d0b056e730d5/Random_1_trials"
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