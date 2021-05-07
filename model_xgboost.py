"""Modélisation catboost"""
import pandas as pd
from xgboost import XGBRegressor

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

xgb_model = XGBRegressor(max_depth=10,
                         n_estimators=1000,
                         min_child_weight=0.5,
                         colsample_bytree=0.8,
                         subsample=0.8,
                         eta=0.1,
                         seed=42)
xgb_model.fit(X_train,
              y_train,
              eval_metric="rmse",
              eval_set=[(X_train, y_train),
                        (X_valid, y_valid)],
              verbose=20,
              early_stopping_rounds=20)

#  Prédiction
xgb_val_pred = xgb_model.predict(X_valid).clip(0, 20)
xgb_test_pred = xgb_model.predict(X_test).clip(0, 20)

#  Export
xgb_val_pred.to_csv('data/catboost_val.csv')
submission = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": xgb_test_pred
})
submission.to_csv('data/xgboost_submission.csv', index=False)
