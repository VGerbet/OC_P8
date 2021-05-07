"""Modélisation catboost"""
import pandas as pd
import lightgbm as lgb

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

params = {'metric': 'rmse',
          'num_leaves': 255,
          'learning_rate': 0.005,
          'feature_fraction': 0.75,
          'bagging_fraction': 0.75,
          'bagging_freq': 5,
          'force_col_wise' : True,
          'random_state': 10}

dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_valid, y_valid)

lgb_model = lgb.train(params=params,
                      train_set=dtrain,
                      num_boost_round=1500,
                      valid_sets=(dtrain, dvalid),
                      early_stopping_rounds=150,
                      categorical_feature=cat_cols,
                      verbose_eval=100)

#  Prédiction
lgb_val_pred = lgb_model.predict(X_valid).clip(0, 20)
lgb_test_pred = lgb_model.predict(X_test).clip(0, 20)

#  Export
lgb_val_pred.to_csv('data/lightgbm_val.csv')
submission = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": lgb_test_pred
})
submission.to_csv('data/lightgbm_submission.csv', index=False)
