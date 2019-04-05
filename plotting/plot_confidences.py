import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import pandas as pd
import pickle
import json


def plot(df, strategy_name, plot_name, dim, save_figure=False):
    print("Graphing results...")
    plt.figure(figsize=(5, 4))
    print(strategy_name[0], strategy_name[1])
    df["Strategy_del"] = strategy_name[0]
    df2 = df
    print(df2)
    df2["Strategy"] = strategy_name[1]
    fig = sns.pointplot(x='Trial', y='Probability_x',
                        data=df, hue="Strategy_del"
                        )
    sns.pointplot(x='Trial', y='Probability_y',
                    data=df2, palette="hls", hue="Strategy"
                        )
    sns.set_context("notebook", font_scale=1)
    fig.set(ylabel="Posterior Probability")
    fig.set(xlabel="Trial Number")
    fig.set(title="BAMS on %s dimension manipulating %s" % (dim, manipulate))
    fig.xaxis.get_major_ticks()
    fig.xaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.legend(loc='upper left')
    if save_figure == True:
        plt.savefig("/Users/darpa/Downloads/%s.png" % "BAMS on dimension %s" % (dim))
    else:
        plt.show()


def open_trial_data(path, strategy_name, save_figure=False):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    model['Predictor'] = 'human'
    print(model)
    fig = sns.pointplot(x="n_dots", y="guess", data=model, hue="Predictor");
    sns.set_context("notebook", font_scale=1)
    fig.set(ylabel="Number of dots predicted")
    fig.set(xlabel="Number of dots presented")
    fig.set(title=strategy_name)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    if save_figure == True:
        fig.savefig("/Users/darpa/Downloads/%s.png" % strategy_name)
    else:
        plt.show()


def translate_file(s):
    s = str(s).split()[0]
    translate = s.replace("(", "_").rstrip(',')
    return translate


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


def plot_top_5_models(root_path, strategy):
    pickle_path = root_path + "/" + strategy
    df = pd.read_pickle("%s.pkl" % pickle_path)
    print(df.tail(1).sort_values)
    df_names = pd.DataFrame({'list_of_models': list(df)})
    series = df[df.iloc[-5:].idxmax(axis=1)]
    print(df[df.columns])


def model_predict(plot_path, val):
    print(plot_path)
    model = pickle.load(open(plot_path), 'rb')
    return model.predict(val)


mapId = "mapping_2d99"
with open("../mappings/%s.json" % mapId) as json_file:
    mapping = json.load(json_file)

dim = 1
manipulate = "dots"
root = "../data/"
strategies = ["BALD_%s" % manipulate, "Random_%s" % manipulate]
BALD_PATH_ROOT = root + mapping["BALD_" + manipulate]
BALD_PATH_ALL = BALD_PATH_ROOT + "/all_models"
RANDOM_PATH_ROOT = root + mapping["Random_" + manipulate]
RANDOM_PATH_ALL = RANDOM_PATH_ROOT + "/all_models"

df1, plot_name1 = get_best_model_and_name(BALD_PATH_ROOT, strategies[0])
df2, plot_name2 = get_best_model_and_name(RANDOM_PATH_ROOT, strategies[1])
#plot_top_5_models(BALD_PATH_ROOT, strategies[0])
#sys.exit()

plot_names = [plot_name1, plot_name2]
merged_df = pd.merge(df1, df2, on='Trial')

#plot_path1 = ("%s/%s/%s") % (BALD_PATH_ALL, strategies[0], translate_file(plot_names[0]))
#plot_path2 = ("%s/%s/%s") % (RANDOM_PATH_ALL, strategies[1], translate_file(plot_names[1]))
#open_trial_data(BALD_PATH_ROOT + "/" + strategies[0] + "_trials.pkl", strategies[0])# This opens trial data
print("BALD %s converges to" % str(manipulate))
print(plot_name1)
print("Random %s converges to" % str(manipulate))
print(plot_name2)
plot(merged_df, list(strategies), list(plot_names), str(dim), save_figure=False)

