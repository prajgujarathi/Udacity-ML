import pandas as pd
import numpy as np
import pickle




if __name__ == '__main__':
    dtest = pd.read_pickle("processed/df_test.pkl")
    features_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
                       'user_average_days_between_orders', 'user_average_basket',
                       'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
                       'aisle_id', 'department_id', 'product_orders', 'product_reorders',
                       'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
                       'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last','UP_delta_hour_vs_last'
                       ]
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
    filename = "Xgboost_model.sav"
    alg = pickle.load(open(filename, 'rb'))
    dtest_predictions = alg.predict(dtest[features_to_use])
    dtest_predprob = alg.predict_proba(dtest[features_to_use])[:, 1]

    dtest['pred'] = dtest_predprob
    TRESHOLD = 0.15
    #select only those product which has probability greater than threshold which is calculated by CV on train set

    d = dict()
    for row in dtest.itertuples():
        if row.pred > TRESHOLD:
            try:
                d[row.order_id] += ' ' + str(row.product_id)
            except:
                d[row.order_id] = str(row.product_id)

    for order in test_orders.order_id:
        if order not in d:
            d[order] = 'None'
    for key in d:
        if bool(d[key]) == False:
            d[key] = 'None'




    sub = pd.DataFrame.from_dict(d, orient='index')

    sub.reset_index(inplace=True)
    sub.columns = ['order_id', 'products']
    sub.to_csv('final_predication_xgb.csv', index=False)
