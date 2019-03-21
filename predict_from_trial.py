from bams.learners import ActiveLearner
from bams.query_strategies import (
    BALD,
    HyperCubePool,
    RandomStrategy,
)
import seaborn as sns; sns.set()
import pandas as pd


NDIM = 1
POOL_SIZE = 3
BUDGET = 4
BASE_KERNELS = ["PER", "LIN", "K"]
DEPTH = 1

mapping = {
            "fake_human_BALD_1": "fake_human/86480bd4-af58-42f3-8582-bdaee9b0b40a", #new,
            "fake_human_random_1": "fake_human/50e384f3-761a-49e4-acaa-a9089ea970fd",
            "BALD_1": "9d1f44fb-51ec-4ea1-a022-56c4607111d5",  # abbr
            "Random_1": "d549103a-e963-4d79-b755-65d1cee29ff7",  # abbr
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


def train_learner(learner, df):
    for i in range(0, len(df)):
        guess = df["guess"].loc[i] / 100.0
        correct = df["n_dots"].loc[i] / 100.0
        learner.update(correct, guess)
        print(correct, guess)
        print(i)
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
BALD_PATH = root + "%s/BALD_1_trials" % mapping["BALD_1"]
RANDOM_PATH = root + "%s/Random_1_trials" % mapping["Random_1"]
bald_out = root + "%s/BALD_1_predictions_all.pkl" % mapping["BALD_1"]
random_out = root + "%s/Random_1_predictions_all.pkl" % mapping["Random_1"]
print(BALD_PATH)
bald_df = pd.read_pickle("%s.pkl" % BALD_PATH)
random_df = pd.read_pickle("%s.pkl" % RANDOM_PATH)
x = [x for x in range(1,100)]

# Script begins
print(bald_df)
print(random_df)

# Training and Predictions
train_learner(bald_learner, bald_df)
train_learner(random_learner, random_df)
bald_prediction_100 = run_predictions(bald_learner)
random_prediction_100 = run_predictions(random_learner)

# Prepare data
save_bald = pd.DataFrame(data={"x":x,"y":bald_prediction_100})
save_bald['Predictor'] = 'BALD Strategy'
save_random = pd.DataFrame(data={"x":x,"y":random_prediction_100})
save_random['Predictor'] = 'RANDOM_Strategy'

print(bald_prediction_100)
print(random_prediction_100)

# Save predictions
save_bald.to_pickle(bald_out)
save_random.to_pickle(random_out)
