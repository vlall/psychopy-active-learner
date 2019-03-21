import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import pandas as pd
import pickle


def plot(df, strategy_name, plot_name, dim, save_figure=True):
    print("Graphing results...")
    plt.figure(figsize=(5, 4))
    print(strategy_name[0], strategy_name[1])
    df["Strategy_del"] = strategy_name[0]
    df2 = df
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
    fig.set(title="BAMS on dimension %s" % (dim))
    fig.xaxis.get_major_ticks()
    fig.xaxis.set_major_locator(ticker.MultipleLocator(10))
    fig.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.legend(loc='upper left')
    if save_figure == True:
        plt.savefig("/Users/darpa/Downloads/%s.png" % "BAMS on dimension %s" % (dim))
    plt.show()


def open_trial_data(path, strategy_name):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    model['Predictor'] = 'human'
    fig = sns.lmplot(x="n_dots", y="guess", data=model, hue="Predictor");
    sns.set_context("notebook", font_scale=1)
    fig.set(ylabel="Number of dots predicted")
    fig.set(xlabel="Number of dots presented")
    fig.set(title=strategy_name)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    fig.savefig("/Users/darpa/Downloads/%s.png" % strategy_name)
    #plt.show()


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


def model_predict(plot_path, val):
    print(plot_path)
    model = pickle.load(open(plot_path), 'rb')
    return model.predict(val)


mapping = {
            "BALD_1":"5b38c816-8789-4749-8843-eae10283f6e2/", # 200 pool
            "BALD_2":"448b79b2-b6d3-41a3-9d94-596647fb84c7", # 200 pool
            "BALD_3":"a9ad42f2-e3e6-453d-a8d2-85200bb15efd", # 200 pool
            "BALD_All":"25e92109-e50c-40ea-aee7-939d7864bbed", # 200 pool
            "Random_1":"c7f82681-9701-4675-b6d5-1ae42729f73c", # 200 pool
            "Random_2":"4a32ce59-0a3e-4cbc-947f-ca52ff682271", # 200 pool
            "Random_3":"56722bfc-4b17-4bb4-acd3-d0197cfc1590", # 200 pool
            "Random_All":"ebc78678-538d-44b9-8292-03d397c20b6c", # 200 pool
            "fake_human_BALD_1": "dummy_oracle/8d2323cf-3292-41cc-8e35-cf1cdefa61f8", # testing,
            "fake_human_random_1": "fake_human/50e384f3-761a-49e4-acaa-a9089ea970fd", # testing
}
dim = "1"
strategies = ["BALD_%s" % str(dim), "Random_%s" % str(dim)]
BALD_PATH_ROOT = mapping["fake_human_BALD_1"]
BALD_PATH_ALL = BALD_PATH_ROOT + "/all_models"
RANDOM_PATH_ROOT = mapping["fake_human_random_1"]
RANDOM_PATH_ALL = RANDOM_PATH_ROOT + "/all_models"

df1, plot_name1 = get_best_model_and_name(BALD_PATH_ROOT, strategies[0])
df2, plot_name2 = get_best_model_and_name(RANDOM_PATH_ROOT, strategies[1])

plot_names = [plot_name1, plot_name2]
merged_df = pd.merge(df1, df2, on='Trial')

#plot_path1 = ("%s/%s/%s") % (BALD_PATH_ALL, strategies[0], translate_file(plot_names[0]))
#plot_path2 = ("%s/%s/%s") % (RANDOM_PATH_ALL, strategies[1], translate_file(plot_names[1]))
open_trial_data(BALD_PATH_ROOT + "/" + strategies[0] + "_trials.pkl", strategies[0])# This opens trial data
print("BALD****")
print(plot_name1)
print("Random****")
print(plot_name2)
plot(merged_df, list(strategies), list(plot_names), dim)
