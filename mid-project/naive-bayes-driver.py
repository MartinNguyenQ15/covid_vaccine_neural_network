import pandas as pd
import numpy as np
from naive_bayes import naive_bayes_gaussian, naive_bayes_cat
from visualize import show_heatmap, show_histogram
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

print("Naive Bayes Probability")
print("=" * 12)
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

print("\t\tGaussian Naive Bayes")
train, test = train_test_split(without, test_size=.2, random_state=41)

X_test = test.iloc[:,1:].values
Y_test = test.iloc[:,0].values
Y_pred = naive_bayes_gaussian(train, X=X_test, Y="county")
# Fix for confusion matrix comparisons
Y_test = np.array(Y_test).astype(int)
Y_pred = Y_pred.astype(int)

nb_gaussian_cm = confusion_matrix(y_pred=Y_pred, y_true=Y_test)
nb_gaussian_f1 = f1_score(Y_test, Y_pred)

print("\tGaussian Distribution -> Confusion Matrix")
print(nb_gaussian_cm)
print("\tGaussian Distribution -> F1 Score")
print(nb_gaussian_f1)
print("=" * 15)
print("\t\tCategorical Naive Bayes")
catgauss = without
catgauss['cat_pfizer_doses'] = pd.cut(catgauss['pfizer_doses'].values, bins=5, labels=[x for x in range(5)])
catgauss['cat_moderna_doses'] = pd.cut(catgauss['moderna_doses'].values, bins=5, labels=[x for x in range(5)])
catgauss['cat_jj_doses'] = pd.cut(catgauss['jj_doses'].values, bins=5, labels=[x for x in range(5)])

train_cat, test_cat = train_test_split(catgauss, test_size=.2, random_state=41)
Xcat_test = test_cat.iloc[:, 1:].values
Ycat_test = test_cat.iloc[:, 0].values

Ycat_pred = naive_bayes_cat(catgauss, X=Xcat_test, Y='county')

Ycat_test = np.array(Ycat_test).astype(int)
Ycat_pred = Ycat_pred.astype(int)

nb_cat_cm = confusion_matrix(y_pred=Ycat_pred, y_true=Ycat_test)
nb_cat_f1 = f1_score(Ycat_test, Ycat_pred)

print("\tCategorical -> Confusion Matrix")
print(nb_cat_cm)
print("\tCategorical -> F1 Score")
print(nb_cat_f1)

