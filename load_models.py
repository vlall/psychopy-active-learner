import numpy as np
import pickle
import uuid
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import random


def run_predictions(learner, manipulate):
    """ Use input as a number """
    prediction_100 = []
    for i in range(0, 100):
        print("Input:" + str(i))
        if manipulate == "all":
            two = random.random()
            three = random.random()
            output = learner.predict([[float(i / 100.0), two, three]])[0]
        else:
            output = learner.predict([float(i / 100.0)])[0]
        print(output)
        prediction_100.append(output[0] * 100)
    return (prediction_100)


def plot(df, manipulate):
    print(df)
    fig = sns.scatterplot(x="manipulation", y="prediction", data=df, hue="Predictor")
    fig.set(ylabel="Number of dots predicted")
    if manipulate == "dots" or manipulate == "all":
        fig.set(xlabel="Number of dots presented")
    else:
        fig.set(xlabel="Level of %s variable" % manipulate)
    plt.show()


def get_best_model_and_name(root_path, strategy):
    pickle_path = root_path + "/" + strategy
    df = pd.read_pickle("%s.pkl" % pickle_path)
    print(df.tail(1).sort_values)
    df_names = pd.DataFrame({'list_of_models': list(df)})
    plot_name = df[df.iloc[-1:].idxmax(axis=1).iloc[0]].name
    series = df[df.iloc[-1:].idxmax(axis=1).iloc[0]]
    df = series.to_frame(name=None)
    df.columns = ['Probability']
    df['Trial'] = range(1, len(df) + 1)
    df['Strategy'] = strategy
    df['Model_name'] = plot_name
    return df, plot_name


def translate(model):
    s = str(model).split()[0]
    translate = s.replace("(", "_").rstrip(',')
    return translate


def main():
    mapId = "mapping_10c9"
    with open("mappings/%s.json" % mapId) as json_file:
        mapping = json.load(json_file)
    strategy = "Random"
    manipulate = "contrast"
    file_name = "%s_%s" % (strategy, manipulate)
    if not os.path.exists("data/%s" % mapping[file_name]):
        print("No such file '{}'".format(mapping[file_name]), file=sys.stderr)
    ROOT_PATH = "data/%s/all_models/" % uuid
    PATH = ROOT_PATH + file_name
    df, plot_name = get_best_model_and_name("data/%s" % uuid, file_name)
    NAME = translate(plot_name)
    print(plot_name)
    FULL_PATH = "%s/%s.pkl" % (PATH, NAME)
    print(FULL_PATH)
    with open(FULL_PATH, 'rb') as f:
        learner = pickle.load(f)
    if manipulate == "dots" or manipulate == "all":
        x = [x for x in range(0, 100)]
    else:
        x = list(np.arange(0.0, 1.0, 0.01))
    y = run_predictions(learner, manipulate)
    bald_df = pd.DataFrame(data={"manipulation": x, "prediction": y})
    bald_df['Predictor'] = file_name
    plot(bald_df, manipulate)


if __name__ == "__main__":
    main()
