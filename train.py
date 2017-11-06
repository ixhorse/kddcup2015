#-*-coding:utf-8-*-

import xgboost as xgb
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from itertools import zip_longest
import time
import merge

class train:
    def __init__(self):
        start_time = time.time()
        #self.enft = enrollment_feature.EnrollmentFT()
        label = self.get_label()
        train = self.get_train()
        data_time = int((time.time() - start_time)/60)

        X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.3, random_state=1)
        xgb_train = xgb.DMatrix(X_train, y_train)
        xgb_test = xgb.DMatrix(X_test, y_test)

        params = {
            'booster': 'gbtree',
            'objective': 'multi:softmax',  # 多分类的问题
            'num_class': 2,  # 类别数，与 multisoftmax 并用
            'gamma': 0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            'max_depth': 7,  # 构建树的深度，越大越容易过拟合
            'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            'subsample': 0.7,  # 随机采样训练样本
            'colsample_bytree': 0.7,  # 生成树时进行的列采样
            'min_child_weight': 3,
            # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
            # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
            # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
            'eta': 0.008,  # 如同学习率
            'seed': 1000,
            'nthread': 12,  # cpu 线程数
            # 'eval_metric': 'auc'
        }
        plist = list(params.items())
        num_rounds = 6000
        model = xgb.train(plist, xgb_train, num_rounds, evals=[(xgb_test, 'eval'), (xgb_train, 'train')], early_stopping_rounds=100)
        pred = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)

        c_num = 0
        for y, y_t in zip_longest(y_test, pred):
            #print(y, y_t, type(y), type(y_t))
            if int(y) == int(y_t):
                c_num += 1
        print(c_num / len(pred))
        print(data_time, ((time.time() - start_time) / 60))


    def get_label(self):
        label = []
        with open('../data/train/truth_train.csv') as f:
            info = csv.reader(f)
            for row in info:
                en_id, truth = row
                #if en_id in self.enft.user_course:
                label.append(truth)
        return np.array(label)

    def get_train(self):
        return np.array(merge.merge().features)

t = train()