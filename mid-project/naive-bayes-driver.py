import pandas as pd
from naive_bayes import naive_bayes_gaussian
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

# sns.set_style("darkgrid")

df = pd.read_csv('./cali_dataset.csv')

county_list = [
    'Calaveras',
    'Lassen'
]

selected_columns = ['county', 'total_doses', 'cumulative_total_doses']

without = df.loc[df['county'].isin(county_list)]
for i, county in enumerate(county_list):
    without.loc[df['county'] == county, 'county'] = i

without = without[selected_columns]
without = without.head(10)
print(without)
# corr = without.corr(method="pearson")
# cmap = sns.diverging_palette(250, 354, 80, 60, center='dark', as_cmap=True)
# sns.heatmap(corr, vmax=1, vmin=-.5, cmap=cmap, square=True, linewidths=.2)
# fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
# sns.histplot(without, ax=axes[0], x="total_doses", kde=True, color='r')
# sns.histplot(without, ax=axes[1], x="cumulative_total_doses", kde=True, color='b')
# plt.show()
train, test = train_test_split(without, test_size=.2, random_state=41)

X_test = test.iloc[:,:-1].values
Y_test = test.iloc[:,:-1].values
Y_pred = naive_bayes_gaussian(train, X=X_test, Y="county")
print(confusion_matrix(Y_test, Y_pred))
print(f1_score(Y_test, Y_pred))
