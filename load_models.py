import numpy as np
import pickle
import uuid
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

def run_predictions(learner):
    prediction_100 = []
    for dots in range(1, 100):
        print("Input:" + str(dots))
        output = learner.predict([float(dots / 100.0)])[0]
        print(output)
        prediction_100.append(output[0] * 100)
    return (prediction_100)


def plot(df):
    print(df)
    fig = sns.scatterplot(x="n_dots", y="prediction", data=df, hue="Predictor")
    fig.set(ylabel="Number of dots predicted")
    fig.set(xlabel="Number of dots presented")
    plt.show()


def main():
    PATH = "data/feeb770c-5ad2-4132-90fa-aea723aa0c95/all_models/BALD_1/"
    NAME = "LocalGaussianKernel_location=3.9590976522410446.pkl"
    FULL_PATH = PATH + NAME

    with open(FULL_PATH, 'rb') as f:
        learner = pickle.load(f)
    x = [x for x in range(1, 100)]
    y = run_predictions(learner)
    bald_df = pd.DataFrame(data={"n_dots": x, "prediction": y})
    bald_df['Predictor'] = 'BALD Strategy'
    plot(bald_df)

if __name__ == "__main__":
    main()