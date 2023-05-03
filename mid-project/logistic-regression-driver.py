import numpy as np
from loaders import load_by_year, load_by_county
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

without, feature_range = load_by_county()
print(without)
X = without.iloc[:, 1:].values
y = without.iloc[:,0].values

scaler = MinMaxScaler(feature_range=feature_range)
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42, )

logreg = LogisticRegression()
logreg.fit(X_train, y_train, epochs=500, lr=0.5)
y_pred = logreg.predict(X_test)

y_test = np.array(y_test).astype("float32")
y_pred = np.array(y_pred).astype("float32")
print("\t\tLogistic Regression")
print("\tLogistic Regression -> Classification Report")
print(classification_report(y_test, y_pred))
print("\tLogistic Regression -> Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
