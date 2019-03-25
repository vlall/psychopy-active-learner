from bams.learners import ActiveLearner
from bams.query_strategies import (
    BALD,
    HyperCubePool,
    RandomStrategy,
)
import seaborn as sns; sns.set()
import pandas as pd


NDIM = 1
POOL_SIZE = 200
BUDGET = 50
BASE_KERNELS = ["PER", "LIN", "K"]
DEPTH = 2

mapping = {
            "fake_human_BALD_1": "fake_human/86480bd4-af58-42f3-8582-bdaee9b0b40a", #new,
            "fake_human_random_1": "fake_human/50e384f3-761a-49e4-acaa-a9089ea970fd",
            "BALD_1": "6b18151d-b50b-4d10-bd1a-058d50f30748",  # abbr
            "Random_1": "9c702a06-ce75-4b1a-b853-e76ad16c2377",  # abbr
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

def train_learner(learner, df, winning_likelihood, linear_likelihood):
    print("There are %d models." % len(df))
    for i in range(0, len(df)):
        searchTerm = "LinearKernel"
        for model in learner.models:
            if str(model).split("(")[0] == searchTerm:
                linear = model
                break
        #linear_list.append(linear.log_likelihood())
        guess = df["guess"].loc[i] / 100.0
        correct = df["n_dots"].loc[i] / 100.0
        learner.update(correct, guess)
        print(i)
        winning_likelihood.append(learner.map_model.log_likelihood())
        print(learner.map_model)
        print(learner.map_model.log_likelihood())
        print(linear)
        print(linear.log_likelihood())
        linear_likelihood.append(linear.log_likelihood())
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

# Training
train_learner(bald_learner, bald_df, bald_likelihood, bald_linear)
train_learner(random_learner, random_df, random_likelihood, random_linear)

# Predictions
bald_prediction_100 = run_predictions(bald_learner)
random_prediction_100 = run_predictions(random_learner)

# Transform prediction data
#save_bald = pd.DataFrame(data={"x":x,"y":bald_prediction_100})
#save_bald['Predictor'] = 'BALD Strategy'
#save_random = pd.DataFrame(data={"x":x,"y":random_prediction_100})
#save_random['Predictor'] = 'RANDOM_Strategy'

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
     })
likelihood_df.to_pickle(likelihood_out)

#Print/Save Predictions
#print(bald_prediction_100)
#print(random_prediction_100)
#save_bald.to_pickle(bald_out)
#save_random.to_pickle(random_out)
