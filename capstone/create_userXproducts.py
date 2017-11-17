
import numpy as np
import pandas as pd
import os

if __name__=='__main__':
    priors = pd.read_csv("order_products__prior.csv",dtype={'order_id':np.int32,
                                                           'product_id':np.uint16,
                                                           'add_to_cart_order':np.int16,
                                                           'reordered':np.int8} )
    orders = pd.read_csv("orders.csv",dtype={'order_id':np.int32,
                                             'user_id':np.int32,
                                             'eval_set':'category',
                                             'order_number':np.uint8,
                                             'order_dow':np.uint8,
                                             'order_hour_of_day':np.uint8,
                                             'days_since_prior_order':np.float64
                                             })
    products = pd.read_csv( "products.csv", dtype={'product_id': np.uint16,
                                                                      'aisle_id': np.uint8,
                                                                      'department_id': np.uint8})
    print('add order info to priors')
    orders.set_index('order_id', inplace=True, drop=False)
    priors = priors.join(orders, on='order_id', rsuffix='_')
    priors.drop('order_id_', inplace=True, axis=1)
    print('compute userXproduct f - this is long...')
    priors['user_product'] = priors.product_id + priors.user_id * 100000

    # This was to slow !!
    # def last_order(order_group):
    #    ix = order_group.order_number.idxmax
    #    return order_group.shape[0], order_group.order_id[ix],  order_group.add_to_cart_order.mean()
    # userXproduct = pd.DataFrame()
    # userXproduct['tmp'] = df.groupby('user_product').apply(last_order)

    d = dict()
    for row in priors.itertuples():
        z = row.user_product
        if z not in d:
            d[z] = (1,
                    (row.order_number, row.order_id),
                    row.add_to_cart_order)
        else:
            d[z] = (d[z][0] + 1,
                    max(d[z][1], (row.order_number, row.order_id)),
                    d[z][2] + row.add_to_cart_order)

    print 'to dataframe (less memory)'
    userXproduct = pd.DataFrame.from_dict(d, orient='index')
    del d
    userXproduct.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart']
    userXproduct.nb_orders = userXproduct.nb_orders.astype(np.int16)
    userXproduct.last_order_id = userXproduct.last_order_id.map(lambda x: x[1]).astype(np.int32)
    userXproduct.sum_pos_in_cart = userXproduct.sum_pos_in_cart.astype(np.int16)
    print('user X product f', len(userXproduct))
    print userXproduct.head()
    userXproduct.fillna(0)
    del priors
    userXproduct.to_pickle('processed/userXproduct_features.pkl')
