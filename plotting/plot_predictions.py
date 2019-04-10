import numpy as np
import matplotlib.patches as mpatches
import pickle
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

mapping = {
    "fake_human_BALD_1": "fake_human/86480bd4-af58-42f3-8582-bdaee9b0b40a",  # new,
    "fake_human_random_1": "fake_human/50e384f3-761a-49e4-acaa-a9089ea970fd",
    "BALD_1": "6b18151d-b50b-4d10-bd1a-058d50f30748",  # abbr
    "Random_1": "9c702a06-ce75-4b1a-b853-e76ad16c2377",  # abbr
}

title = "BALD"
dim = "1"
BALD_PATH = "%s/%s/" % ("data", mapping["BALD_1"])
RANDOM_PATH = "%s/%s/" % ("data", mapping["Random_1"])
bald_predict_df = pd.read_pickle(BALD_PATH + "BALD_%s_predictions_all.pkl" % dim)
random_predict_df = pd.read_pickle(RANDOM_PATH + "Random_%s_predictions_all.pkl" % dim)
bald_human = "%s/BALD_%s_trials" % (BALD_PATH, dim)
random_human = "%s/Random_%s_trials" % (RANDOM_PATH, dim)
bald_df = pd.read_pickle("%s.pkl" % bald_human)
random_df = pd.read_pickle("%s.pkl" % random_human)
print(bald_predict_df)
print(random_predict_df)
print(bald_df)
print(random_df)
x = [x for x in range(1, 100)]

bald_df['Predictor'] = 'Human'
random_df['Predictor'] = 'Human'
bald_predict_df['Predictor'] = 'BALD_Strategy'
random_predict_df['Predictor'] = 'Random_Strategy'

if title.lower() == "random":
    fig = sns.scatterplot(x="n_dots", y="guess", data=random_df, hue="Predictor")
    fig = sns.pointplot(x="x", y="y", data=random_predict_df, hue="Predictor", palette="hls")

elif title.lower() == "bald":
    fig = sns.scatterplot(x="n_dots", y="guess", data=bald_df, hue="Predictor")
    fig = sns.pointplot(x="x", y="y", data=bald_predict_df, hue="Predictor", palette="hls", s=.1)

"""
fig = sns.scatterplot(x="n_dots", y="guess", data=df, )
fig = sns.pointplot(x="x", y="y", data=random_learner_df, hue="Predictor")
fig = sns.scatterplot(x="n_dots", y="guess", data=df3, )
fig = sns.pointplot(x="x", y="y", data=BALD_learner_df, hue="Predictor", palette="hls",s=.1)
"""

# matplotlib attempt
# fig = df3.plot(x="n_dots", y="guess", style=".")
# fig = plt.plot(x=[x for x in range(1,100)], y=f2, color="blue")

# sns.set_context("notebook", font_scale=1)

# fig, ax = plt.subplots()
# fig = plt.plot(x=[x for x in range(1,100)], y=f)
# fig = plt.scatter(x="n_dots", y="guess", data=model)

leg_handles = fig.get_legend_handles_labels()[0]
handles, labels = fig.get_legend_handles_labels()
fig.legend(handles=handles[1:], labels=labels[1:])
fig.set(ylabel="Number of dots predicted")
fig.set(xlabel="Number of dots presented")
xticks = fig.xaxis.get_major_ticks()
fig.xaxis.set_major_locator(ticker.MultipleLocator(10))
fig.xaxis.set_major_formatter(ticker.ScalarFormatter())
fig.set(title="%s on %s" % (title, dim))
# plt.xlim(0, 100)
# plt.ylim(0, 100)
plt.show()
