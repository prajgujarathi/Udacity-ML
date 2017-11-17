import numpy as np
import pandas as pd
import os

if __name__=='__main__':
    order_prior = pd.read_csv("order_products__prior.csv",dtype={'order_id':np.int32,
                                                           'product_id':np.uint16,
                                                           'add_to_cart_order':np.int16,
                                                           'reordered':np.int8} )
    orders = pd.read_csv('orders.csv',dtype={'order_id':np.int32,
                                             'user_id':np.int32,
                                             'eval_set':'category',
                                             'order_number':np.uint8,
                                             'order_dow':np.uint8,
                                             'order_hour_of_day':np.uint8,
                                             'days_since_prior_order': np.float32})
    orders.set_index('order_id', inplace=True, drop=False)
    order_prior = order_prior.join(orders, on='order_id', rsuffix='_')
    order_prior.drop('order_id_', inplace=True, axis=1)


    print('computing user f')
    usr = pd.DataFrame()
    usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
    usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)
    usr['average_hour_of_day'] = orders.groupby('user_id')['order_hour_of_day'].mean().astype(np.float32)
    usr['average_order_dow'] = orders.groupby('user_id')['order_dow'].mean().astype(np.float32)

    users = pd.DataFrame()
    users['total_items'] = order_prior.groupby('user_id').size().astype(np.int16)
    users['all_products'] = order_prior.groupby('user_id')['product_id'].apply(set)
    users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)

    users = users.join(usr)
    del usr
    users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)
    print('user f', users.shape)

    print 'processed'
    print users.head()
    print 'save'

    users.to_pickle('processed/user_features.pkl')

