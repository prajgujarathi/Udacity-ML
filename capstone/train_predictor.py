
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    # Fit the algorithm on the data
    alg.fit(dtrain, predictors)

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain)
    dtrain_predprob = alg.predict_proba(dtrain)[:, 1]

    # Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, dtrain, predictors, cv=cv_folds,
                                                    scoring='roc_auc',  )

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(predictors.values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(predictors, dtrain_predprob)

    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (
        np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))

    # Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_,dtrain.columns).sort_values(ascending=False)
        print feat_imp
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

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
       'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last'
       ]
    df_train = df_train.dropna(axis=0,how='any')
    features_final = df_train[features_to_use]
    label = df_train['labels']
    del df_train
    X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                        label,
                                                        test_size=0.2,
                                                        random_state=40)

    # Show the results of the split
    print "Training set has {} samples.".format(X_train.shape[0])
    print "Testing set has {} samples.".format(X_test.shape[0])

    param_test1 = {'n_estimators': range(20, 81, 10)}
    gsearch1 = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500, min_samples_leaf=50, max_depth=8,
                                             max_features='sqrt', subsample=0.8, random_state=10),
        param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch1.fit(X_test, y_test)
    print gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_
    #clf_A = KNeighborsClassifier()
    clf_B = GradientBoostingClassifier(random_state=40,)
    clf_C = AdaBoostClassifier(random_state=40)
    # Collect results on the learners
    results = {}
    for clf in [clf_B]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        #modelfit(clf, X_train, y_train)

    # Run metrics visualization for the three supervised learning models chosen
    #vs.evaluate(results, accuracy, fscore)