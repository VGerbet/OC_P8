"""Modélisation catboost"""
import pandas as pd
from catboost import CatBoostRegressor

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

cat_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 31, 32]

catboost_model = CatBoostRegressor(
    iterations=500,
    max_ctr_complexity=4,
    random_seed=0,
    od_type='Iter',
    od_wait=25,
    verbose=50,
    depth=4
)

catboost_model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_valid, y_valid)
)

#  Prédiction
catboost_val_pred = catboost_model.predict(X_valid).clip(0, 20)
catboost_test_pred = catboost_model.predict(X_test).clip(0, 20)

#  Export
catboost_val_pred.to_csv('data/catboost_val.csv')
submission = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": catboost_test_pred
})
submission.to_csv('data/catboost_submission.csv', index=False)
