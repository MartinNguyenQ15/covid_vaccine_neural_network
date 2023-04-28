import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from visualize import show_heatmap

df = pd.read_csv('./cali_dataset.csv')

# Issue to resolve with f1_score if more than 2 cities
county_list = [
    'Calaveras',
    # 'Lassen',
    # 'Alpine',
    'Plumas'
]

# Selecting columns to use for classification
selected_columns = ['county', 'pfizer_doses', 'moderna_doses', 'jj_doses']

without = df.loc[df['county'].isin(county_list)]
for i, county in enumerate(county_list):
    without.loc[df['county'] == county, 'county'] = i

without = without[selected_columns]

show_heatmap(without)
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
sns.histplot(without, ax=axes[0,0], x="pfizer_doses", kde=True, color='r')
sns.histplot(without, ax=axes[0,1], x="moderna_doses", kde=True, color='g')
sns.histplot(without, ax=axes[1,0], x="jj_doses", kde=True, color='b')
plt.show()