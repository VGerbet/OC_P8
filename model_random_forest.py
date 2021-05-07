"""Modélisation catboost"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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

#  Top 10 features XGboost
top_10 = ['item_cnt_lag1', 'new_item', 'item_cnt_trend', 'transaction_nb_lag1',
          'item_cnt_mean_item_lag1', 'item_category_id', 'since_first_sale',
          'item_cnt_mean_shop_cat_lag1', 'month', 'item_cnt_mean_city_lag1']

rf_model = RandomForestRegressor(n_estimators=50,
                                 max_depth=7,
                                 random_state=0,
                                 n_jobs=-1)
rf_model.fit(X_train[top_10], y_train)

#  Prédiction
rf_val_pred = rf_model.predict(X_valid[top_10]).clip(0, 20)
rf_test_pred = rf_model.predict(X_test[top_10]).clip(0, 20)

#  Export
rf_val_pred.to_csv('data/random_forest_val.csv')
submission = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": rf_test_pred
})
submission.to_csv('data/random_forest_submission.csv', index=False)
