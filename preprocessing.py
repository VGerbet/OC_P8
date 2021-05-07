"""This script downloads and preprocess data"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import utils

# Import
sales = pd.read_csv('data/sales_train.csv',
                    parse_dates=["date"])
items = pd.read_csv('data/items.csv')
item_cat = pd.read_csv('data/item_categories.csv')
shops = pd.read_csv('data/shops.csv')
test = pd.read_csv('data/test.csv')


##################################################################
######## PREPARATION #############################################
##################################################################

#  Filtre des outliers
sales = sales.query('item_cnt_day > 0 & item_cnt_day < 1000').copy()
sales = sales.query('item_price > 0 & item_price <= 50000').copy()

#  Cleaning doublons
sales.loc[sales['shop_id'] == 0, 'shop_id'] = 57
sales.loc[sales['shop_id'] == 1, 'shop_id'] = 58
sales.loc[sales['shop_id'] == 10, 'shop_id'] = 11
sales.loc[sales['shop_id'] == 39, 'shop_id'] = 40

test.loc[test['shop_id'] == 0, 'shop_id'] = 57
test.loc[test['shop_id'] == 1, 'shop_id'] = 58
test.loc[test['shop_id'] == 10, 'shop_id'] = 11
test.loc[test['shop_id'] == 39, 'shop_id'] = 40

#  Focus sur les shops présents dans le test set
shop_ids = test['shop_id'].unique()
sales = sales[sales['shop_id'].isin(shop_ids)]

#  Features extraites des noms
shops["city"] = shops.shop_name.apply(lambda x: x.split()[0])
shops["category"] = shops.shop_name.apply(lambda x: x.split()[1])
shops.loc[shops['city'] =='!Якутск', 'city'] = 'Якутск' #  C.f. cleaning

item_cat["type_code"] = item_cat.item_category_name.apply(lambda x: x.split()[0])

#  On ne conserve que les principales
shops_cat = shops.category.value_counts()
thresh_cat = shops_cat[shops_cat >= 5].index
shops.category = shops.category.apply(utils.thresh_filter,
                                      args=([thresh_cat]))
item_types = item_cat.type_code.value_counts()
thresh_type = item_types[item_types >= 5].index
item_cat.type_code = item_cat.type_code.apply(utils.thresh_filter,
                                              args=([thresh_type]))
item_cat["subtype"] = item_cat.item_category_name.apply(utils.get_subtype)

#  Encoding
shops["shop_category"] = LabelEncoder().fit_transform(shops.category)
shops["shop_city"] = LabelEncoder().fit_transform(shops.city)
item_cat.type_code = LabelEncoder().fit_transform(item_cat.type_code)
item_cat["subtype_code"] = LabelEncoder().fit_transform(item_cat['subtype'])

shops = shops[["shop_id", "shop_category", "shop_city"]]
item_cat = item_cat[["item_category_id", "subtype_code", "type_code"]]

#  Première apparition d'un item
items['first_sale_date'] = sales.groupby('item_id')\
                                .agg({'date_block_num': 'min'})['date_block_num']
items['first_sale_date'] = items['first_sale_date'].fillna(34)

#  Aggregation mensuelle et création des enregistrements manquants
groupby_cols = ['date_block_num', 'shop_id', 'item_id']
sales['transaction'] = sales['item_cnt_day'] * sales['item_price']

monthly_sales = sales.groupby(by=groupby_cols,
                              as_index=False).agg({'item_cnt_day': ['sum',
                                                                    'count'],
                                                   'transaction': 'sum',
                                                   'item_price': 'mean',
                                                   })
monthly_sales.columns = ['date_block_num', 'shop_id', 'item_id',
                         'item_cnt', 'transaction_nb', 'transaction', 'mean_price']
full_sales = utils.fill_missing_month(monthly_sales)

##################################################################
######## FEATURE ENGINEERING #####################################
##################################################################

#  Concatenation du jeu de test
test['date_block_num'] = 34
full_sales = pd.concat([full_sales, test.drop('ID', axis=1)],
                       ignore_index=True,
                       keys=groupby_cols)
full_sales = full_sales.fillna(0)

#  Récupération des Features
full_sales = full_sales.merge(shops,
                              on='shop_id',
                              how='left')
full_sales = full_sales.merge(items,
                              on=['item_id'],
                              how='left')
full_sales = full_sales.merge(item_cat,
                              on='item_category_id',
                              how='left')

#  Features temporelles
full_sales['year'] = full_sales.date_block_num.apply(utils.extract_year)
full_sales['month'] = full_sales.date_block_num.apply(utils.extract_month)
full_sales['new_item'] = full_sales['first_sale_date'] == full_sales['date_block_num']
full_sales['since_first_sale'] = full_sales['date_block_num'] - full_sales['first_sale_date']

#  Moyennes mensuelles
full_sales = utils.get_month_mean(full_sales,
                                  ['date_block_num', 'item_category_id', 'shop_id'],
                                  suffixes=('', '_mean_shop_cat'))
full_sales = utils.get_month_mean(full_sales,
                                  ['date_block_num', 'item_id'],
                                  suffixes=('', '_mean_item'))
full_sales = utils.get_month_mean(full_sales,
                                  ['date_block_num', 'item_id', 'shop_city'],
                                  suffixes=('', '_mean_city'))

#  Lag features
lag_list = [1, 2, 3]
utils.get_lag_feature(full_sales, 'item_cnt', ['shop_id', 'item_id'], lag_list)
utils.get_lag_feature(full_sales, 'transaction_nb', ['shop_id', 'item_id'], lag_list)
utils.get_lag_feature(full_sales, 'mean_price', ['shop_id', 'item_id'], lag_list)
utils.get_lag_feature(full_sales, 'item_cnt_mean_city', ['shop_id', 'item_id'], lag_list)
utils.get_lag_feature(full_sales, 'item_cnt_mean_item', ['shop_id', 'item_id'], lag_list)
utils.get_lag_feature(full_sales, 'item_cnt_mean_shop_cat', ['shop_id', 'item_id'], lag_list)

#  Clipping
cnt_cols = []
for col in full_sales.columns:
    if '_cnt' in col:
        cnt_cols.append(col)
for col in cnt_cols:
    full_sales[col] = full_sales[col].clip(0, 20)

#  Trends
full_sales['item_cnt_trend'] = full_sales[['item_cnt_lag1',
                                           'item_cnt_lag2',
                                           'item_cnt_lag3']].mean(axis=1)
full_sales['item_cnt_trend'].fillna(0, inplace=True)
full_sales['trend1'] = full_sales['item_cnt_lag1'] / full_sales['item_cnt_lag2']
full_sales['trend1'] = full_sales['trend1'].replace([np.inf, -np.inf], np.nan)
full_sales['trend1'] = full_sales['trend1'].fillna(0)

full_sales['trend2'] = full_sales['item_cnt_lag2'] / full_sales['item_cnt_lag3']
full_sales['trend2'] = full_sales['trend2'].replace([np.inf, -np.inf], np.nan)
full_sales['trend2'] = full_sales['trend2'].fillna(0)

#  Finalisation du jeu de données
full_sales = full_sales.query('date_block_num >= 3').copy()
droped_col = ['transaction_nb', 'transaction', 'mean_price', 'item_name',
               'first_sale_date', 'item_cnt_mean_shop_cat', 'item_cnt_mean_item',
              'item_cnt_mean_city',]
full_sales.drop(columns=droped_col, inplace=True)

#  Export
full_sales.to_csv('data/processed_data.csv')
