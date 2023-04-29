import matplotlib.pyplot as plt
import seaborn as sns
from loaders import load_by_year, load_by_county
from visualize import show_heatmap

without, range = load_by_county()
show_heatmap(without, 'Vaccine usage by counties Calaveras and Plumas, CA')
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
axes[0,0].set(xlabel=" ", title="Pfizer dose usage")
axes[0,1].set(xlabel=" ", title="Moderna dose usage")
axes[1,0].set(xlabel=" ", title="JJ dose usage")
sns.histplot(without, ax=axes[0,0], x="pfizer_doses", kde=True, color='r')
sns.histplot(without, ax=axes[0,1], x="moderna_doses", kde=True, color='g')
sns.histplot(without, ax=axes[1,0], x="jj_doses", kde=True, color='b')
# ax = plt.axes()
# ax.set_title('Vaccine usage by maker company')
fig.suptitle("Vaccine usage by maker company in Calaveras and Plumas, CA")
plt.show()