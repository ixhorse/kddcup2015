#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math

import numpy as np
from sklearn.model_selection import GridSearchCV

sys.path.append('xgboost/wrapper/')
import xgboost as xgb
import csv
from sklearn.model_selection import train_test_split
from itertools import zip_longest
import time
import merge


class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)

    def predict(self, X):
        num2label = {i: label for label, i in self.label2num.items()}
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self


def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in
                    zip(Y_pred, y_true)) / len(Y_pred)

def get_label():
    label = []
    with open('../data/train/truth_train.csv') as f:
        info = csv.reader(f)
        for row in info:
            en_id, truth = row
            #if en_id in self.enft.user_course:
            label.append(truth)
    return np.array(label)

def get_train():
    return np.array(merge.merge().features)


def main():
    clf = XGBoostClassifier(
        eval_metric='auc',
        num_class=2,
        nthread=10,
        silent=1,
    )
    parameters = {
        'num_boost_round': [500, 1000, 2500, 5000],
        'eta': [0.05, 0.1, 0.3],
        'max_depth': [6, 9, 12],
        'subsample': [0.9, 1.0],
        'colsample_bytree': [0.9, 1.0],
    }
    clf = GridSearchCV(clf, parameters, n_jobs=-1, cv=2)

    start_time = time.time()
    label = get_label()
    train = get_train()
    data_time = int((time.time() - start_time) / 60)

    X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.3, random_state=1)
    xgb_train = xgb.DMatrix(X_train, y_train)
    xgb_test = xgb.DMatrix(X_test, y_test)


    clf.fit(train, label)
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == '__main__':
    main()