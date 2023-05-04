import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

df = pd.read_csv('./cali_dataset.csv')
meta_columns = ['county', 'administered_date']
data_columns = ['total_doses']
# data_columns = ['total_doses','cumulative_total_doses','pfizer_doses', 'jj_doses', 'moderna_doses', 'total_partially_vaccinated']
selected_columns = []
selected_columns.extend(meta_columns)
selected_columns.extend(data_columns)

without = df[selected_columns]

without = without[without['county'] == 'Calaveras']
without = without.drop(columns=[meta_columns[0]])
without = without[::-1]

without = without.loc[without['administered_date'].str.contains(r'|'.join(['2022']), regex=True)]
for enu_i, rp in enumerate(without.iterrows()):
    idx, row = rp
    without.loc[idx, 'administered_date'] = enu_i + 1

features = without.drop(columns=data_columns, axis=1)
labels = without.drop(columns=[meta_columns[1]], axis=1)
print(features)
print(labels)
print(without[99:130])

# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

X_train = np.array(features.values).astype('int')
# X_test = np.array(X_test.values).astype('int')
y_train = np.array(labels.values).astype('int')
# y_test = np.array(y_test.values).astype('int')

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    Dense(units=y_train.shape[1]), # Prediction of the following set of doses
])

model.compile(optimizer='adam', loss='msle')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
y_pred = model.predict([[125]])
print("Pred:", y_pred)

