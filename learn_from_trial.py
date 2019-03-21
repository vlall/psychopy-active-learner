import bams
import pandas as pd
import cPickle as pickle
from bams.learners import ActiveLearner
from bams.query_strategies import (
    BALD,
    HyperCubePool,
    RandomStrategy,
)
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker
import pylab
import cPickle as pickle
import matplotlib.patches as mpatches
import bams.learners
import numpy as np

NDIM = 1
POOL_SIZE = 25
BUDGET = 50
BASE_KERNELS = ["PER", "LIN", "K"]
DEPTH = 2

mapping = {
            "fake_human_BALD_1": "fake_human/86480bd4-af58-42f3-8582-bdaee9b0b40a", #new,
            "fake_human_random_1": "fake_human/50e384f3-761a-49e4-acaa-a9089ea970fd",
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

path2 = "%s/BALD_1_trials" % mapping["fake_human_BALD_1"]
path = "%s/Random_1_trials" % mapping["fake_human_random_1"]
out2 = "%s/BALD_1_predictions_all.pkl" % mapping["fake_human_BALD_1"]
out1 = "%s/Random_1_predictions_all.pkl" % mapping["fake_human_random_1"]
df3 = pd.read_pickle("%s.pkl" % path2)
df = pd.read_pickle("%s.pkl" % path)
x = [x for x in range(1,100)]
print df3

# Read the dimiensions and set learner to len(ndim)
# feed learned the dimensions?
# Return dots/100
i=0


def run_predictions(learner):
    pass

for i in xrange(0,random_learner.budget):
    guess = df3["guess"].loc[i] / 100.0
    correct = df3["n_dots"].loc[i] / 100.0
    random_learner.update(correct, guess)
    print(correct, guess)
    print(i)
    i+=1

f = []
for dots in xrange(1,100):
    print("Input:" + str(dots))
    output = random_learner.predict([float(dots/100.0)])[0]
    f.append(output[0]*100)
    print(output)

i = 0
# BALD
for i in xrange(0, bald_learner.budget):
    guess = df3["guess"].loc[i] / 100.0
    correct = df3["n_dots"].loc[i] / 100.0
    bald_learner.update(correct, guess)
    print(correct, guess)
    print(i)
    i+=1
f2 = []
for dots in xrange(1,100):
    print("Input:" + str(dots))
    output = bald_learner.predict([float(dots/100.0)])[0]
    f2.append(output[0]*100)
    print(output)
# END BALD

#df['Predictor'] = 'RANDOM_Human'
df2 = pd.DataFrame(data={"x":x,"y":f})
df2['Predictor'] = 'Random Strategy'
# fig = sns.scatterplot(x="n_dots", y="guess", data=df, hue="Predictor")
# fig = sns.pointplot(x=x, y=f, data=df2, hue="Predictor", palette="hls")

#BALD
#df3['Predictor'] = 'BALD_Human'
print(df3)
print(f2)
df4 = pd.DataFrame(data={"x":x,"y":f2})
df4['Predictor'] = 'BALD_Strategy'
#fig = sns.scatterplot(x="n_dots", y="guess", data=df3, hue="Predictor")
#fig = sns.pointplot(x=[x for x in range(1,100)], y=f2, data=df4, hue="Predictor", palette="hls",s=.1)

#SAVE PREDICTIONS
df2.to_pickle(out1)
df4.to_pickle(out2)

# matplotlib attempt
#fig = df3.plot(x="n_dots", y="guess", style=".")
#fig = plt.plot(x=[x for x in range(1,100)], y=f2, color="blue")

sns.set_context("notebook", font_scale=1)

#fig, ax = plt.subplots()
#fig = plt.plot(x=[x for x in range(1,100)], y=f)
#fig = plt.scatter(x="n_dots", y="guess", data=model)
# seaborn uncomment
# leg_handles = fig.get_legend_handles_labels()[0]
# handles, labels = fig.get_legend_handles_labels()
# fig.legend(handles=handles[1:], labels=labels[1:])
# fig.set(ylabel="Number of dots predicted")
# fig.set(xlabel="Number of dots presented")
# xticks=fig.xaxis.get_major_ticks()
# fig.xaxis.set_major_locator(ticker.MultipleLocator(10))
# fig.xaxis.set_major_formatter(ticker.ScalarFormatter())
# fig.set(title="Random Learner on Dimension 1")
# plt.xlim(0, 100)
# plt.ylim(0, 100)
# plt.show()