import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.regularizers import l2

df = pd.read_csv('./cali_dataset.csv')
selected_columns = ['county', 'administered_date', 'pfizer_doses', 'jj_doses', 'moderna_doses']
without = df[selected_columns]

without = without[without['county'] == 'Calaveras']
without = without.drop(columns=['county'])
without = without.loc[without['administered_date'].str.contains(''.join(['2022']), regex=False)]
features = without.drop(columns=['pfizer_doses', 'jj_doses', 'moderna_doses'], axis=1)
labels = without.drop(columns=['administered_date'], axis=1)

features['administered_date'] = pd.to_datetime(features['administered_date'])
features['administered_date'] = features['administered_date'].apply(lambda x: x.toordinal())

X = pd.get_dummies(features)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    Dense(units=3), # Prediction of the following set of doses
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)
y_pred = model.predict(X_test)
print(X_test)
print(y_pred)
# print("Pred:", y_pred[:10])
# print("Actual", y_test[:10])

# loss = model.evaluate(X_test, y_test)

# predicted_rates = model.predict(X_test)

# num_sims = 10

# sim_results = np.zeros(num_sims)
# alpha = 6.985

# for i in range(num_sims):
#     vac_rates = np.random.normal(predicted_rates.mean(), predicted_rates.std(), size=len(X_test))
#     vac_counts = vac_rates * alpha
#     sim_results[i] = vac_counts.sum()

# mean = sim_results.mean()
# std = sim_results.std()

# print(mean)
# print(sum(y_pred))
# print(sim_results)