import pandas as pd
from bams.learners import ActiveLearner
from bams.query_strategies import (
    BALD,
    HyperCubePool,
    RandomStrategy,
)
import seaborn as sns
sns.set()


NDIM = 1
POOL_SIZE = 200
BUDGET = 20
BASE_KERNELS = ["PER", "LIN", "K"]
DEPTH = 2

mapping = {"BALD_dots": "2198efd7-2724-4322-9c3b-fba4f4cf9b32", "Random_dots": "d39a7b20-6ce5-49b4-9597-48555184d528",
     "BALD_contrast": "2e71966c-f655-4a90-bd44-66a53725fec1", "Random_contrast": "7c7c5563-5698-46d3-a34b-22e3e03466b9",
     "BALD_random": "6ef4c5c4-5eae-4b1c-a93b-ad0615a1770e", "Random_random": "58298351-9a47-4312-89aa-c2be4fcf657d",
     "BALD_all": "32eade98-f87c-4c62-a889-4b3d5b0ff587", "Random_all": "8dbe1445-9b2c-46f2-badb-dc491bd3c9de"
}

random_learner = ActiveLearner(
    query_strategy=RandomStrategy(pool=HyperCubePool(NDIM, POOL_SIZE)),
    budget=BUDGET,
    base_kernels=BASE_KERNELS,
    max_depth=DEPTH,
    ndim=NDIM,
)

bald_learner = ActiveLearner(
    query_strategy=BALD(pool=HyperCubePool(NDIM, POOL_SIZE)),
    budget=BUDGET,
    base_kernels=BASE_KERNELS,
    max_depth=DEPTH,
    ndim=NDIM,
)

bald_likelihood = []
random_likelihood = []
bald_linear = []
random_linear = []
bald_constant = []
random_constant = []


def train_learner(learner, df, winning_likelihood, linear_likelihood, constant_likelihood):
    print("There are %d models." % len(df))
    for i in range(0, len(df)):
        lineTerm = "LinearKernel"
        constantTerm = "ConstantKernel"
        for model in learner.models:
            splitTerm = str(model).split("(")[0]
            if "+" in splitTerm or "*" in splitTerm:
                continue
            elif splitTerm == lineTerm:
                linear = model
            elif splitTerm == constantTerm:
                constant = model
        guess = df["guess"].loc[i] / 100.0
        correct = df["n_dots"].loc[i] / 100.0
        learner.update(correct, guess)
        print(i)
        winning_likelihood.append(learner.map_model.log_likelihood())
        print(learner.map_model)
        print(learner.map_model.log_likelihood())
        print(linear)
        print(linear.log_likelihood())
        print(constant)
        print(constant.log_likelihood())
        print(learner.map_model.entropy(1))
        linear_likelihood.append(linear.log_likelihood())
        constant_likelihood.append(constant.log_likelihood())
    return True


def run_predictions(learner):
    prediction_100 = []
    for dots in range(1, 100):
        print("Input:" + str(dots))
        output = random_learner.predict([float(dots / 100.0)])[0]
        print(output)
        prediction_100.append(output[0] * 100)
    return (prediction_100)


root = "data/"
BALD_PATH = root + "%s/BALD_contrast_trials" % mapping["BALD_contrast"]
RANDOM_PATH = root + "%s/Random_contrast_trials" % mapping["Random_contrast"]
bald_out = root + "%s/BALD_contrast_predictions_all.pkl" % mapping["BALD_contrast"]
random_out = root + "%s/Random_contrast_predictions_all.pkl" % mapping["Random_contrast"]
print(BALD_PATH)
bald_df = pd.read_pickle("%s.pkl" % BALD_PATH)
random_df = pd.read_pickle("%s.pkl" % RANDOM_PATH)
x = [x for x in range(1, 100)]

# Script begins
print(bald_df)
print(random_df)

# Training
train_learner(bald_learner, bald_df, bald_likelihood, bald_linear, bald_constant)
train_learner(random_learner, random_df, random_likelihood, random_linear, random_constant)

# Predictions
bald_prediction_100 = run_predictions(bald_learner)
random_prediction_100 = run_predictions(random_learner)

# Transform prediction data
save_bald = pd.DataFrame(data={"x":x,"y":bald_prediction_100})
save_bald['Predictor'] = 'BALD Strategy'
save_random = pd.DataFrame(data={"x":x,"y":random_prediction_100})
save_random['Predictor'] = 'RANDOM_Strategy'

# Save likelihoods of models
likelihood_out = root + "/likelihood-pickle.pkl"
print(random_linear)
print(random_likelihood)
print(bald_linear)
print(bald_likelihood)
likelihood_df = pd.DataFrame(
    {'Random_Linear': random_linear,
     'Random_Likelihood': random_likelihood,
     'BALD_Linear': bald_linear,
     'BALD_Likelihood': bald_likelihood,
     'BALD_constant': bald_constant,
     'Random_constant': random_constant,
     })
likelihood_df.to_pickle(likelihood_out)

# Print/Save Predictions
# print(bald_prediction_100)
# print(random_prediction_100)
save_bald.to_pickle(bald_out)
save_random.to_pickle(random_out)
