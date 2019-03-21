import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker
import pickle
import matplotlib.patches as mpatches
import numpy as np


mapping = {
            "fake_human_BALD_1": "fake_human/86480bd4-af58-42f3-8582-bdaee9b0b40a", #new,
            "fake_human_random_1": "fake_human/50e384f3-761a-49e4-acaa-a9089ea970fd",
            "BALD_1": "474ed916-5bb4-45f7-ad41-ed8273f3f766",  # abbr
            "Random_1": "f39f200f-bde3-47ac-83c5-c8df6fde9485",  # abbr
}

dim = "1"
BALD_PATH = "%s/%s/" % ("data", mapping["BALD_1"])
RANDOM_PATH = "%s/%s/" % ("data", mapping["Random_1"])
BALD_learner_df = pd.read_pickle(BALD_PATH + "BALD_%s_predictions_all.pkl" % dim)
random_learner_df = pd.read_pickle(RANDOM_PATH + "Random_%s_predictions_all.pkl" % dim)
BALD_human = "%s/BALD_%s_trials" % (BALD_PATH, dim)
random_human = "%s/Random_%s_trials" % (RANDOM_PATH, dim)
df3 = pd.read_pickle("%s.pkl" % BALD_human)
df = pd.read_pickle("%s.pkl" % random_human)
print(df3)
print(df)
print(BALD_learner_df)
print(random_learner_df)
x = [x for x in range(1,100)]

df['Predictor'] = 'Human'
df3['Predictor'] = 'Human'
random_learner_df['Predictor'] = 'Random_Strategy'
BALD_learner_df ['Predictor'] = 'BALD_Strategy'

#fig = sns.scatterplot(x="n_dots", y="guess", data=df, hue="Predictor")
#fig = sns.pointplot(x="x", y="y", data=random_learner_df, hue="Predictor", palette="hls")


fig = sns.scatterplot(x="n_dots", y="guess", data=df3, hue="Predictor")
fig = sns.pointplot(x="x", y="y", data=BALD_learner_df, hue="Predictor", palette="hls",s=.1)

"""
fig = sns.scatterplot(x="n_dots", y="guess", data=df, )
fig = sns.pointplot(x="x", y="y", data=random_learner_df, hue="Predictor")
fig = sns.scatterplot(x="n_dots", y="guess", data=df3, )
fig = sns.pointplot(x="x", y="y", data=BALD_learner_df, hue="Predictor", palette="hls",s=.1)
"""

leg_handles = fig.get_legend_handles_labels()[0]
handles, labels = fig.get_legend_handles_labels()
fig.legend(handles=handles[1:], labels=labels[1:])
fig.set(ylabel="Number of dots predicted")
fig.set(xlabel="Number of dots presented")
xticks=fig.xaxis.get_major_ticks()
fig.xaxis.set_major_locator(ticker.MultipleLocator(10))
fig.xaxis.set_major_formatter(ticker.ScalarFormatter())
fig.set(title="Random on %s" % dim)
#plt.xlim(0, 100)
#plt.ylim(0, 100)
plt.show()