import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Data
df = pd.read_pickle('data/likelihood-pickle.pkl')
x = [x for x in range(1,51)]
df["Trials"] = x
print(df)
# multiple line plot
df = df.melt('Trials', var_name='cols',  value_name='Log Likelihood')
g = sns.factorplot(x="Trials", y="Log Likelihood", hue='cols', data=df)
plt.legend()
plt.show()