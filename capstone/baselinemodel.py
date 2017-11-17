from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier  # GBM algorithm
from sklearn import cross_validation, metrics
from sklearn.metrics import make_scorer, precision_recall_curve
# Additional scklearn functions
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import pickle
from collections import Counter


rcParams['figure.figsize'] = 12, 4


def precision_at_recall(y_true, y_score, constraint):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return np.max(precision[recall >= constraint])

def modelfit(alg, dtrain, predictors,dtest,ytest,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    # Fit the algorithm on the data
    '''if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain.values, label=predictors)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    print 'cvresult'
    print cvresult

    alg.set_params(n_estimators=55)

        # Fit the algorithm on the data
    alg.fit(dtrain, predictors, eval_metric='auc')
    filename = "Xgboost_model.sav"
    pickle.dump(alg, open(filename, 'wb'))'''
    # Predict training set


    filename = "Xgboost_model.sav"
    alg = pickle.load(open(filename, 'rb'))
    dtrain_predictions = alg.predict(dtrain)
    dtrain_predprob = alg.predict_proba(dtrain)[:, 1]

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    print feat_imp
    print "Predicting Train set probability!"
    dtrain_predprob = alg.predict_proba(dtrain, )[:, 1]
    dtest_predictions = alg.predict(dtest)
    dtest_predprob = alg.predict_proba(dtest)[:, 1]

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(predictors, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(predictors, dtrain_predprob)
    print "F1 score: %f" % metrics.fbeta_score(predictors, dtrain_predictions, average='binary', beta=0.5)

    print "Accuracy : %.4g" % metrics.accuracy_score(ytest, dtest_predictions)
    print "AUC Score (Test): %f" % metrics.roc_auc_score(ytest, dtest_predprob)
    print "F1 score: %f" % metrics.fbeta_score(ytest, dtest_predictions,beta=0.5)


    precision, recall, thresholds = precision_recall_curve(predictors, dtrain_predprob)
    print  np.max(precision[recall >= 0.95])
    #precision_at_recall_score = make_scorer(precision_at_recall(np.max(precision[recall >= 0.95]), needs_threshold=True, constraint=0.95)


if __name__ == '__main__':
    df_train = pd.read_pickle("processed/df_train.pkl")
    features_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
                       'user_average_days_between_orders', 'user_average_basket',
                       'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
                       'aisle_id', 'department_id', 'product_orders', 'product_reorders',
                       'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
                       'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last','UP_delta_hour_vs_last'
                       ]
    df_train = df_train.dropna(axis=0, how='any')
    features_final = df_train[features_to_use]
    label = df_train['labels']

    print Counter(label).keys()  # equals to list(set(words))
    print Counter(label).values()

    dtrain_predictions = [0] * df_train.shape[0]
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(label, dtrain_predictions)
    print "F1 score: %f" % metrics.fbeta_score(label, dtrain_predictions, average='binary', beta=0.5)

    TP = label.count() - np.sum(label)  # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data
    # encoded to numerical values done in the data preprocessing step.
    FP = label.count() - TP  # Specific to the naive case

    TN = 0  # No predicted negatives in the naive case
    FN = 0  # No predicted negatives in the naive case

    print TP
    print FP
    print label.count()
    # TODO: Calculate accuracy, precision and recall

    accuracy = float(TP) / float(label.count())
    recall = 1
    precision = accuracy
    beta = 0.5

    # TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
    # HINT: The formula above can be written as (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    fscore = (1 + 0.25) * (precision * recall) / ((0.25 * precision) + recall)

    # Print the results
    print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)


    print Counter
    del df_train
