
import numpy as np
import pandas as pd
import os

def organic(prod):
    if 'organic' in prod.lower():
        return 1
    else:
        return 0

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
                                             'days_since_prior_order':np.float32
                                             })
    products = pd.read_csv( "products.csv", dtype={'product_id': np.uint16,
                                                                      'aisle_id': np.uint8,
                                                                      'department_id': np.uint8})

    print 'loaded'
    prd = pd.DataFrame()
    prd['orders'] = order_prior.groupby('product_id').size().astype(np.int32)
    prd['reorders'] = order_prior.groupby('product_id')['reordered'].sum().astype(np.float32)
    prd['reorder_rate'] = (prd['reorders'] / prd['orders']).astype(np.float32)
    #prd['order_rate'] = prd['orders'] / prd.count()
    products = products.join(prd,on='product_id')

    #products['organic'] = products.product_name.apply(organic)
    products.set_index('product_id',drop=False,inplace=True)
    del prd

    if not os.path.isdir('processed'):
        os.makedirs('processed')
    print 'processed'
    print products.head()
    print 'save'
    products.fillna(0)
    products.to_pickle('processed/product_features.pkl')
