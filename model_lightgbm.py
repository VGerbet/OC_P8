"""Modélisation catboost"""
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
import numpy as np



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

cat_cols = ['shop_id', 'item_id', 'year', 'month', 'item_category_id',
            'shop_category', 'shop_city', 'subtype_code', 'type_code', 'new_item']

params = dict(metric=['rmse'],
              num_leaves=[10, 31, 255, 400],
              learning_rate=[0.005, 0.001],
              max_depth=[-1],
              feature_fraction=[0.75],
              bagging_fraction=[0.75],
              bagging_freq=[5],
              random_state=[10],
              verbose=[-1])

dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_valid, y_valid)

#  GridSearch
grid_results = {}
for i, param in enumerate(ParameterGrid(params)):
    grid_results[i] = {}
    grid_results[i]['params'] = param
    curr_model = lgb.train(params=param,
                           train_set=dtrain,
                           num_boost_round=1000,
                           valid_sets=(dtrain, dvalid),
                           early_stopping_rounds=150,
                           categorical_feature=cat_cols,
                           verbose_eval=False)
    grid_results[i]['model'] = curr_model
    pred = curr_model.predict(X_valid)
    val_rmse = np.sqrt(mean_squared_error(pred, y_valid))
    grid_results[i]['val_rmse'] = val_rmse

#  Meilleure itération
best_iter = sorted(grid_results,
                   key=lambda k: grid_results[k]['val_rmse'])[0]
best_model = grid_results[best_iter]['model']

#  Prédiction
lgb_val_pred = best_model.predict(X_valid).clip(0, 20)
lgb_test_pred = best_model.predict(X_test).clip(0, 20)

#  Export
lgb_val_pred.to_csv('data/lightgbm_val.csv')
submission = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": lgb_test_pred
})
submission.to_csv('data/lightgbm_submission.csv', index=False)
