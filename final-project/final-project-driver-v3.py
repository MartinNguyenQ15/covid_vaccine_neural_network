import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, Dropout, Normalization

df = pd.read_csv('./cali_dataset.csv')
meta_columns = ['county', 'administered_date', 'fully_vaccinated', 'cumulative_at_least_one_dose']
data_columns = ['total_doses','cumulative_total_doses','pfizer_doses', 'jj_doses', 'moderna_doses', 'total_partially_vaccinated', 'cumulative_fully_vaccinated', 'cumulative_booster_recip_count']
selected_columns = []
selected_columns.extend(meta_columns)
selected_columns.extend(data_columns)

without = df

without = without[without['county'] == 'Calaveras']
without = without.drop(columns=[meta_columns[0]])
without = without[::-1]

# without = without.loc[without['administered_date'].str.contains(r'|'.join(['2022', '2021', '2020']), regex=True)]
for enu_i, rp in enumerate(without.iterrows()):
    idx, row = rp
    without.loc[idx, 'administered_date'] = enu_i

f_list = ['total_doses','pfizer_doses','moderna_doses','jj_doses','partially_vaccinated','fully_vaccinated','at_least_one_dose','booster_recip_count','bivalent_booster_recip_count','booster_eligible_population','bivalent_booster_eligible_population']
l_list = ['cumulative_total_doses']

features = without[f_list]
print(features)

# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

features_train = np.asarray(features.values)
labels_train = np.asarray(without[l_list].values)
print(labels_train)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(features_train)

X_train, X_valtest, y_train, y_valtest = train_test_split(scaled, labels_train, test_size=0.3)
X_val, X_test, y_val, Y_test = train_test_split(X_valtest, y_valtest, test_size=0.5)

model = Sequential([
    Dense(128, input_shape=(len(f_list),), activation='relu'),
    Dropout(0.15),
    Dense(128, activation='relu'),
    Dropout(0.15),
    Dense(54, activation='relu'),
    Dropout(0.18),
    Dense(1),
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))
# y_pred = model.predict([[108, 64746, 1838]])
# print("Pred:", y_pred)
# print(model.evaluate(X_test, Y_test)[1])

