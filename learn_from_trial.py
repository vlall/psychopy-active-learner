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
import seaborn as sns
import pandas as pd


NDIM = 1
POOL_SIZE = 25
BUDGET = 50
BASE_KERNELS = ["PER", "LIN", "K"]
DEPTH = 2

learner = ActiveLearner(
    query_strategy=RandomStrategy(pool=HyperCubePool(NDIM, POOL_SIZE)),
    budget=BUDGET,
    base_kernels=BASE_KERNELS,
    max_depth=DEPTH,
    ndim=NDIM,
)

learner2 = ActiveLearner(
    query_strategy=BALD(pool=HyperCubePool(NDIM, POOL_SIZE)),
    budget=BUDGET,
    base_kernels=BASE_KERNELS,
    max_depth=DEPTH,
    ndim=NDIM,
)


mapping = {
            "fake_human_BALD_1": "fake_human/86480bd4-af58-42f3-8582-bdaee9b0b40a", #new,
            "fake_human_random_1": "fake_human/50e384f3-761a-49e4-acaa-a9089ea970fd",
}
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
for i in xrange(0,learner.budget):
    guess = df3["guess"].loc[i] / 100.0
    correct = df3["n_dots"].loc[i] / 100.0
    learner.update(correct, guess)
    print(correct, guess)
    print(i)
    i+=1

f = []
for dots in xrange(1,100):
    print("Input:" + str(dots))
    output = learner.predict([float(dots/100.0)])[0]
    f.append(output[0]*100)
    print(output)


# BALD
for i in xrange(0,learner2.budget):
    guess = df3["guess"].loc[i] / 100.0
    correct = df3["n_dots"].loc[i] / 100.0
    learner2.update(correct, guess)
    print(correct, guess)
    print(i)
    i+=1
f2 = []
for dots in xrange(1,100):
    print("Input:" + str(dots))
    output = learner2.predict([float(dots/100.0)])[0]
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
