"""Modélisation linéaire"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/processed_data.csv')
test = pd.read_csv('data/test.csv')

#  Jeux d'entraînement, de validation et de test
X_train = data.query('date_block_num < 33')
X_train = X_train.drop(columns=['item_cnt'])

X_valid = data.query('date_block_num == 33')
X_valid = X_valid.drop(columns=['item_cnt'])

X_test = data.query('date_block_num == 34')
X_test = X_test.drop(columns=['item_cnt'])


y_train = data.query('date_block_num < 33').item_cnt
y_valid = data.query('date_block_num == 33').item_cnt

#  Features décorrelées
lr_features = ['date_block_num', 'month','item_cnt_lag1', 'item_cnt_lag2', 'item_cnt_lag3',
               'mean_price_lag1', 'mean_price_lag2', 'mean_price_lag3',
               'item_cnt_mean_city_lag1', 'item_cnt_mean_city_lag2',
               'item_cnt_mean_city_lag3', 'item_cnt_mean_item_lag1',
               'item_cnt_mean_item_lag2', 'item_cnt_mean_item_lag3',
               'item_cnt_mean_shop_cat_lag1', 'item_cnt_mean_shop_cat_lag2',
               'item_cnt_mean_shop_cat_lag3', 'item_cnt_trend', 'lag_grad1',
               'lag_grad2', 'since_first_sale']
lr_train = X_train[lr_features]
lr_val = X_valid[lr_features]
lr_test = X_test[lr_features]

#  Scaling
lr_scaler = StandardScaler()
lr_scaler.fit(lr_train)
lr_train = lr_scaler.transform(lr_train)
lr_val = lr_scaler.transform(lr_val)
lr_test = lr_scaler.transform(lr_test)

#  Entraînement
lr_model = LinearRegression(n_jobs=-1)
lr_model.fit(lr_train, y_train)

#  Prédiction
lr_val_pred = lr_model.predict(lr_val).clip(0, 20)
lr_test_pred = lr_model.predict(lr_test).clip(0, 20)

#  Export
lr_val_pred.to_csv('data/linear_val.csv')
submission = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": lr_test_pred
})
submission.to_csv('data/linear_submission.csv', index=False)
