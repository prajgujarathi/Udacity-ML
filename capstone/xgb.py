from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier  # GBM algorithm
from sklearn import cross_validation, metrics
from sklearn.metrics import make_scorer, precision_recall_curve,confusion_matrix
# Additional scklearn functions
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import pickle


rcParams['figure.figsize'] = 12, 4


def precision_at_recall(y_true, y_score, constraint):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return np.max(precision[recall >= constraint])

def modelfit(alg, dtrain, predictors,dtest,ytest,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    # Fit the algorithm on the data
    if useTrainCV:
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
    pickle.dump(alg, open(filename, 'wb'))
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

    matrix =  confusion_matrix(predictors, dtrain_predictions)
    print matrix
    TP =  matrix[0][0]
    FP = matrix[0][1]  # Specific to the naive case

    TN = matrix[1][1] # No predicted negatives in the naive case
    FN = matrix[1][0] # No predicted negatives in the naive case

    print TP
    print FP
    print label.count()
    # TODO: Calculate accuracy, precision and recall

    accuracy = metrics.accuracy_score(predictors, dtrain_predictions)
    recall = float(TP )/  float(TP + FN)
    precision = float(TP )/ float(TP + FP)
    print recall, precision ,(1 + 0.25) * (precision * recall) ,((0.25 * precision) + recall)
    beta = 0.5

    # TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
    # HINT: The formula above can be written as (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    fscore = (1 + 0.25) * (precision * recall) / ((0.25 * precision) + recall)

    # Print the results
    print "XGB Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)
    precision, recall, thresholds = precision_recall_curve(predictors, dtrain_predprob)
    print  np.max(precision[recall >= 0.95])
    #precision_at_recall_score = make_scorer(precision_at_recall(np.max(precision[recall >= 0.95]), needs_threshold=True, constraint=0.95)



'''def train_predict(learner, X_train, y_train, X_test, y_test):

    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set


    results = {}
    start = datetime.now()   # Get start time
    learner = learner.fit(X_train,)
    end = datetime.now()  # Get end time

    # Calculate the training time
    results['train_time'] = end - start

    # Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = datetime.now()   # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = datetime.now()   # Get end time
    print "predictions_test"
    print predictions_test
    print "y_test"
    print y_test.shape
    # Calculate the total prediction time
    results['pred_time'] = end - start

    # Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    # Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, 0.5)

    # Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5,average='weighted')
    print fbeta_score(y_test, predictions_test, 0.5)
    # Success
    print "{} trained ".format(learner.__class__.__name__, )
    print "f_test: ".format(results['f_test'])
    print "pred_time: ".format(results['pred_time'])
    print "acc_train: ".format(results['acc_train'])
    print "acc_test: ".format(results['acc_test'])
    print "f_train: ".format(results['f_train'])
    print learner.feature_importances_
    # Return the results
    return results
'''
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
    del df_train
    X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                        label,
                                                        test_size=0.1,
                                                        random_state=40)

    # Show the results of the split
    print "Training set has {} samples.".format(X_train.shape[0])
    print "Testing set has {} samples.".format(X_test.shape[0])

    '''param_test1 = {
        'min_child_weight': range(1, 6, 2)
    }'''
    '''param_test3 = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }'''
    param_test4 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    '''gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.5, n_estimators=140, max_depth=9,
                                                    min_child_weight=1, gamma=0.2, subsample= 0.9, colsample_bytree=0.6,
                                                    objective='binary', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test4, scoring='roc_auc', n_jobs=4, iid=False, cv=5)'''
    #gsearch1.fit(X_test, y_test)
    #print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    xgb1 = XGBClassifier(learning_rate=0.2, n_estimators=140, max_depth=9,
                                                    min_child_weight=1, gamma=0.2, subsample= 0.9, colsample_bytree=0.6,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27)
    modelfit(xgb1,X_train,y_train, X_test, y_test)

    # clf_A = KNeighborsClassifier()
    '''clf_B = GradientBoostingClassifier(random_state=40, )
    clf_C = AdaBoostClassifier(random_state=40)
    # Collect results on the learners
    results = {}
    for clf in [clf_B]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        # modelfit(clf, X_train, y_train)'''


