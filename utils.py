"""Définition des fonctions pour le prétraitement des données"""
import pandas as pd

def thresh_filter(string_value,
                  items,
                  default="other"):
    """Renvoie une valeur par défaut si l'élément n'est pas
       dans la liste"""
    return string_value if (string_value in items) else default


def get_subtype(string_value):
    """Sépare une chaîne de caractères et renvoie la deuxième valeur
       si elle existe, sinon la première"""
    split = string_value.split()
    if len(split) > 1:
        res = split[1].strip()
    else:
        res = split[0].strip()
    return res


def fill_missing_month(monthly_sales):
    """Créer les lignes pour les couples (mois, shop, item) manquant

    Args:
        - monthly_sales: DataFrame - ventes mensuelles groupées selon
                         (mois, shop, item)
    Return:
        DataFrame complété
    """
    months_nb = monthly_sales.date_block_num.max()

    full_df = []
    for i in range(months_nb + 1):
        shops = monthly_sales.query('date_block_num == @i').shop_id.unique()
        items = monthly_sales.query('date_block_num == @i').item_id.unique()
        for shop in shops:
            for item in items:
                full_df.append([i, shop, item])
    full_df = pd.DataFrame(full_df,
                           columns=['date_block_num', 'shop_id', 'item_id'])
    full_df = full_df.merge(monthly_sales,
                            how='left',
                            on=['date_block_num', 'shop_id', 'item_id'])
    full_df.fillna(0, inplace=True)

    return full_df


def get_month_mean(full_df,
                   idx_col,
                   suffixes,
                   col='item_cnt'):
    """Gets mean value for each month

    Args:
     - idx_col: columns to group by
     - col: column to groub
    """
    agg_df = full_df[idx_col + [col]].groupby(idx_col).mean()
    agg_df = full_df.merge(agg_df,
                           how='left',
                           on=idx_col,
                           suffixes=suffixes)
    return agg_df


def extract_year(date_num_block, thresh=2013):
    """Extrait l'année à partir d'un nombre de mois
       depuis une référence"""
    return date_num_block // 12 + thresh


def extract_month(date_num_block):
    """Extrait le mois à partir d'un nombre de mois"""
    return date_num_block % 12


def get_lag_feature(full_df,
                    col,
                    idx_col,
                    lag_list):
    """Retrives previous values of col for each value of lag

    Args:
        - col: column of interest
        - idx_col: columns to group by
        - lag_list: intervals of interest"""
    for lag in lag_list:
        ft_name = f'{col}_lag{lag}'
        full_df[ft_name] = full_df.sort_values('date_block_num')\
                                  .groupby(idx_col)[col]\
                                  .shift(lag)
        full_df[ft_name].fillna(0, inplace=True)
