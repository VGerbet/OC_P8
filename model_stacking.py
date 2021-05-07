"""Exploites les résultats de l'ensemble des modèles"""
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data/processed_data.csv')
test = pd.read_csv('data/test.csv')
y_valid = data.query('date_block_num == 33').item_cnt

#  Training
catboost_val_pred = pd.read_csv('data/catboost_val.csv')
lgb_val_pred = pd.read_csv('data/lightgbm_val.csv')
xgb_val_pred = pd.read_csv('data/xgboost_val.csv')
rf_val_pred = pd.read_csv('data/random_forest_val.csv')
lr_val_pred = pd.read_csv('data/linear_val.csv')

first_level = pd.DataFrame(catboost_val_pred, columns=['catboost'])
first_level['lgbm'] = lgb_val_pred
first_level['xgbm'] = xgb_val_pred
first_level['random_forest'] = rf_val_pred
first_level['linear_regression'] = lr_val_pred
first_level['label'] = y_valid.values

meta_model = LinearRegression(n_jobs=-1)
first_level.drop('label', axis=1, inplace=True)
meta_model.fit(first_level, y_valid)
ensemble_pred = meta_model.predict(first_level).clip(0, 20)

#  Testing
catboost_test_pred = pd.read_csv('data/catboost_submission.csv')['item_cnt_month']
lgb_test_pred = pd.read_csv('data/lightgbm_submission.csv')['item_cnt_month']
xgb_test_pred = pd.read_csv('data/xgboost_submission.csv')['item_cnt_month']
rf_test_pred = pd.read_csv('data/random_forest_submission.csv')['item_cnt_month']
lr_test_pred = pd.read_csv('data/linear_submission.csv')['item_cnt_month']

test_data = pd.DataFrame(catboost_test_pred, columns=['catboost'])
test_data['lgbm'] = lgb_test_pred
test_data['xgbm'] = xgb_test_pred
test_data['random_forest'] = rf_test_pred
test_data['linear_regression'] = lr_test_pred

ensemble_test_val = meta_model.predict(test_data).clip(0, 20)
submission = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": ensemble_test_val
})
submission.to_csv('data/stacking_submission.csv', index=False)
