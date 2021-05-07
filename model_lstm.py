"""Modèle LSTM avec keras"""
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

data = pd.read_csv('data/processed_data.csv')
test = pd.read_csv('data/test.csv')

#  Transformation en une série temporelle par couple (shop, item)
monthly_series = data.pivot_table(index=['shop_id', 'item_id'],
                                  columns='date_block_num',
                                  values='item_cnt',
                                  fill_value=0).reset_index()

#  Jeu de test
test_series = test.merge(monthly_series,
                         on=['shop_id', 'item_id'])
test_series.drop(columns=['ID', 'shop_id', 'item_id',
                          'date_block_num', 3, 34],
                          inplace=True)

monthly_series = monthly_series.drop(['item_id', 'shop_id', 34], axis=1)
monthly_series.rename(columns={33: 'label'},
                      inplace=True)

#  Entraînement et validation
labels = monthly_series['label']
monthly_series.drop(columns=['label'], axis=1, inplace=True)
train, valid, Y_train, Y_valid = train_test_split(monthly_series,
                                                  labels.values,
                                                  test_size=0.10,
                                                  random_state=0)
X_train = train.values.reshape((train.shape[0], train.shape[1], 1))
X_valid = valid.values.reshape((valid.shape[0], valid.shape[1], 1))
X_test = test_series.values.reshape((test_series.shape[0],
                                     test_series.shape[1],
                                     1))

#  Modèle
SERIE_SIZE = X_train.shape[1]
N_FEATURES = X_train.shape[2]

EPOCHS = 20
BATCH = 128
LR = 0.0001

lstm_model = keras.Sequential()
lstm_model.add(keras.layers.LSTM(10,
                                 input_shape=(SERIE_SIZE,
                                              N_FEATURES),
                                 return_sequences=True))
lstm_model.add(keras.layers.LSTM(6,
                                 activation='relu',
                                 return_sequences=True))
lstm_model.add(keras.layers.LSTM(1,
                                 activation='relu'))
lstm_model.add(keras.layers.Dense(10,
                                  kernel_initializer='glorot_normal',
                                  activation='relu'))
lstm_model.add(keras.layers.Dense(10,
                                  kernel_initializer='glorot_normal',
                                  activation='relu'))
lstm_model.add(keras.layers.Dense(1))

adam = keras.optimizers.Adam(LR)
lstm_model.compile(loss='mse', optimizer=adam)
lstm_history = lstm_model.fit(X_train, Y_train,
                              validation_data=(X_valid, Y_valid),
                              batch_size=BATCH,
                              epochs=EPOCHS,
                              verbose=2)
lstm_test_pred = lstm_model.predict(X_test).clip(0, 20)
submission = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": lstm_test_pred
})
submission.to_csv('data/lstm_submission.csv', index=False)
