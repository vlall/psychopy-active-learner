import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pylab


def plot(df, strategy_name, plot_name):
    print("Graphing results...")
    plt.figure(figsize=(15, 4))
    fig = sns.pointplot(x='Trial', y='Probability',
                        data=df
                        )
    sns.set_context("notebook", font_scale=1)
    fig.set(ylabel="Posterior Probability")
    fig.set(xlabel="Trial #")
    fig.set(title="%s using %s" % (strategy_name, plot_name))
    for label in fig.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    pylab.show()

#example = {1: 0.014012259115402175, 2: 0.010586020567390508, 3: 0.070410005902961953, 4: 0.065816475737389732, 5: 0.061796272496622742, 6: 0.058218998328712482, 7: 0.055027681609100335, 8: 0.052165106816936604, 9: 0.049583999265996791, 10: 0.047245256172894171, 11: 0.045116550262111407, 12: 0.043170961450528372, 13: 0.041385933650449347, 14: 0.03974244122912219, 15: 0.038222397042778383, 16: 0.03681604865159286, 17: 0.035509363821477707, 18: 0.03429221392687154, 19: 0.033155684547801142, 20: 0.032090499137312531, 21: 0.031092483595154839, 22: 0.030155147403213237, 23: 0.029272645744443545, 24: 0.028440310287794385, 25: 0.027654049843215983, 26: 0.026909789396499701, 27: 0.026205023667561694, 28: 0.025535683697182912, 29: 0.02490017763518234, 30: 0.024295311506569568, 31: 0.023719128912337829, 32: 0.023169637329076512, 33: 0.022645337089835994, 34: 0.022143936557119246, 35: 0.02166425571084064, 36: 0.021204913133697132, 37: 0.020764642167921361, 38: 0.020342279394226982, 39: 0.019936754240742309, 40: 0.019547079648151503, 41: 0.019172344147683828, 42: 0.018811705165993162, 43: 0.01846438176922249, 44: 0.018129650068905916, 45: 0.017806837628315076, 46: 0.017495319051497482, 47: 0.017194511764973924, 48: 0.016903872794889307, 49: 0.016622895168666347, 50: 0.016351105086223491, 51: 0.016088059118251962, 52: 0.015833342095417084, 53: 0.015586564626326321, 54: 0.015347360998596387, 55: 0.015115388187135106, 56: 0.014890323008602387, 57: 0.014671861501209546, 58: 0.014459717161018551, 59: 0.014253620156593261, 60: 0.014053315410616057, 61: 0.013858561984139945, 62: 0.013669132563130739, 63: 0.013484811590952408, 64: 0.013305395068518038, 65: 0.013130690147300715, 66: 0.012960513368896105, 67: 0.012794691035283542, 68: 0.012633058400583314, 69: 0.012475458173566392, 70: 0.012321741722691588, 71: 0.012171767005579141, 72: 0.012025399161009752, 73: 0.011882509634507876, 74: 0.011742975591208302, 75: 0.01160668078003888, 76: 0.011473513294745788, 77: 0.011343366793327196, 78: 0.011216139550206426, 79: 0.011091734711959888, 80: 0.01097005922517522, 81: 0.010851024079179762, 82: 0.010734544565445233, 83: 0.010620539112445105, 84: 0.010508929565915349, 85: 0.010399641586035719, 86: 0.010292603089070516, 87: 0.010187745754044152, 88: 0.010085003026008463, 89: 0.88861227193149717, 90: 0.889713567707764, 91: 0.89079329941842678, 92: 0.89185209458279824, 93: 0.89289055603077983, 94: 0.89390926389273107, 95: 0.89490877672271918, 96: 0.89588963207717442, 97: 0.89685234725545626, 98: 0.89779742090658388, 99: 0.89872533364184037, 100: 0.89963654862063513}
#df = pd.DataFrame(example, columns=['Trial', 'Probablity'])
strategy_name = "BALD 1"
df = pd.read_pickle("%s.pkl" % strategy_name)
plot_name = df[df.iloc[-1:].idxmax(axis=1).iloc[0]].name
series = df[df.iloc[-1:].idxmax(axis=1).iloc[0]]
df = series.to_frame(name=None)
df.columns = ['Probability']
df['Trial'] = range(1, len(df) + 1)
print df
plot(df, strategy_name, plot_name)