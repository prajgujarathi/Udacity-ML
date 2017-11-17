
import pandas as pd
import numpy as np


def features(selected_orders, labels_given=False):
    print 'build candidate list'
    train = pd.read_csv("order_products__train.csv")
    train.set_index(['order_id', 'product_id'], inplace=True, drop=False)
    print "train head"
    print train.head()
    order_list = []
    product_list = []
    labels = []
    i = 0
    users = pd.read_pickle("processed/user_features.pkl")
    print users.head()
    print users.all_products.head()
    products = pd.read_pickle("processed/product_features.pkl")
    print products.head()
    userXproduct = pd.read_pickle("processed/userXproduct_features.pkl")
    print userXproduct.head()

    for row in selected_orders.itertuples():
        i += 1
        if i % 10000 == 0: print('order row', i)
        order_id = row.order_id
        user_id = row.user_id
        user_products = users.all_products[user_id]
        product_list += user_products
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels += [(order_id, product) in train.index for product in user_products]

    df = pd.DataFrame({'order_id': order_list, 'product_id': product_list}, dtype=np.int32)
    labels = np.array(labels, dtype=np.int8)
    del order_list
    del product_list

    print 'user related features'
    df['user_id'] = df.order_id.map(orders.user_id)
    df['user_total_orders'] = df.user_id.map(users.nb_orders)
    df['user_total_items'] = df.user_id.map(users.total_items)
    df['total_distinct_items'] = df.user_id.map(users.total_distinct_items)
    df['user_average_days_between_orders'] = df.user_id.map(users.average_days_between_orders)
    df['user_average_basket'] = df.user_id.map(users.average_basket)

    print 'order related features'
    df['dow'] = df.order_id.map(orders.order_dow)
    df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)
    df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
    df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders

    print 'product related features'
    df['aisle_id'] = df.product_id.map(products.aisle_id)
    df['department_id'] = df.product_id.map(products.department_id)
    df['product_orders'] = df.product_id.map(products.orders).astype(np.int32)
    df['product_reorders'] = df.product_id.map(products.reorders)
    df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)

    print('user_X_product related features')
    df['z'] = df.user_id * 100000 + df.product_id
    df.drop(['user_id'], axis=1, inplace=True)
    df['UP_orders'] = df.z.map(userXproduct.nb_orders)
    df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
    df['UP_average_pos_in_cart'] = (df.z.map(userXproduct.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
    df['UP_reorder_rate'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(orders.order_number)
    df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day - df.UP_last_order_id.map(orders.order_hour_of_day)).map(
        lambda x: min(x, 24 - x)).astype(np.int8)
    df['UP_same_dow_as_last_order'] = df.UP_last_order_id.map(orders.order_dow) == df.order_id.map(orders.order_dow)

    df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)
    print df.dtypes
    print df.memory_usage()
    print df.head()
    print labels
    return df, labels

if __name__ == '__main__':
    orders = pd.read_csv("orders.csv",dtype={'order_id':np.int32,
                                             'user_id':np.int32,
                                             'eval_set':'category',
                                             'order_number':np.uint8,
                                             'order_dow':np.uint8,
                                             'order_hour_of_day':np.uint8,
                                             'days_since_prior_order': np.float32})
    orders.set_index('order_id', inplace=True, drop=False)
    test_orders = orders[orders.eval_set == 'test']
    train_orders = orders[orders.eval_set == 'train']
    df_train, labels = features(train_orders, labels_given=True)
    df_train['labels'] = labels
    df_train.to_pickle("processed/df_train.pkl")
    df_test, labels = features(test_orders)
    df_test.to_pickle("processed/df_test.pkl")

