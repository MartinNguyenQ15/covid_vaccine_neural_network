import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

df = pd.read_csv('./cali_dataset.csv')

without = df.loc[df['county'] == 'Calaveras']
print(without.head(10))
without = without.drop(columns=["county", "administered_date", "california_flag"])
print(without.head(10))
corr = without.corr(method="pearson")
cmap = sns.diverging_palette(250, 354, 80, 60, center='dark', as_cmap=True)
sns.heatmap(corr, vmax=1, vmin=-.5, cmap=cmap, square=True, linewidths=.2)
plt.show()