# import libraries necessary for this project
import os, sys, pickle

import numpy as np
import pandas as pd

from datetime import date

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler

#打开训练和测试数据集
dfoff = pd.read_csv('ccf_offline_stage1_train.csv',keep_default_na=False)
dftest = pd.read_csv('ccf_offline_stage1_test_revised.csv',keep_default_na=False)
# print(dfoff.head(5))

#简单的统计 用户使用优惠券的情况
# print('有优惠卷，购买商品：%d' % dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] != 'null')].shape[0])
# print('有优惠卷，未购商品：%d' % dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] == 'null')].shape[0])
# print('无优惠卷，购买商品：%d' % dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] != 'null')].shape[0])
# print('无优惠卷，未购商品：%d' % dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] == 'null')].shape[0])

# Convert Discount_rate and Distance
"""null--->没有折扣 0--->直接是折扣率  1--->满减"""
def getDiscountType(row):#定义打折类型
    if row == 'null':
        return 'null'
    elif ':' in row:#满减
        return 1
    else:
        return 0#折扣率

def convertRate(row): #统一为折扣率
    """Convert discount to rate"""
    if row == 'null':
        return 1.0
    elif ':' in row:
        rows = row.split(':')
        return 1.0 - float(rows[1])/float(rows[0])
    else:
        return float(row)

def getDiscountMan(row): #返回满多少才可以减的满
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0

def getDiscountJian(row): #返回满多少才可以减的减多少
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0


"""增加4个特征"""
def processData(df):
    # convert discount_rate
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    return df


"""在数据集中添加上面的四个特征"""
dfoff = processData(dfoff)
dftest = processData(dftest)

# convert distance
dfoff['distance'] = dfoff['Distance'].replace('null', -1).astype(int)#-1代替null
dftest['distance'] = dftest['Distance'].replace('null', -1).astype(int)
# print(dfoff['distance'].unique())
# print(dftest['distance'].unique())

# date_received = dfoff['Date_received'].unique()
# date_received = sorted(date_received[date_received != 'null'])
#
# date_buy = dfoff['Date'].unique()
# date_buy = sorted(date_buy[date_buy != 'null'])
#
# print('优惠卷收到日期从',date_received[0],'到',date_received[-1])
# print('消费日期从',date_buy[0],'到',date_buy[-1])

def getWeekday(row):
    if row == 'null':
        return row
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1

dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)

# weekday_type :  周六和周日为1，其他为0
dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x: 1 if x in [6,7] else 0)
dftest['weekday_type'] = dftest['weekday'].apply(lambda x: 1 if x in [6,7] else 0)

# change weekday to one-hot encoding
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
#print(weekdaycols)

tmpdf = pd.get_dummies(dfoff['weekday'].replace('null', np.nan))
tmpdf.columns = weekdaycols
dfoff[weekdaycols] = tmpdf

tmpdf = pd.get_dummies(dftest['weekday'].replace('null', np.nan))
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf

original_feature = ['discount_rate','discount_type','discount_man', 'discount_jian','distance', 'weekday', 'weekday_type'] + weekdaycols
# print('共有特征：',len(original_feature),'个')
# print(original_feature)

def label(row):
    if row['Date_received'] == 'null':
        return -1
    if row['Date'] != 'null':
        td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'): # 规定时间内
            return 1
    return 0

dfoff['label'] = dfoff.apply(label, axis=1)#运行时间比较久,在数据集中添加label标签

# data split
df = dfoff[dfoff['label'] != -1].copy()
train = df[(df['Date_received'] < '20160516')].copy()#选择部分数据
valid = df[(df['Date_received'] >= '20160516') & (df['Date_received'] <= '20160615')].copy()
# print('Train Set: \n', train['label'].value_counts())
# print('Valid Set: \n', valid['label'].value_counts())


def check_model(data, predictors):  # 最关键的 模型建立

    classifier = lambda: SGDClassifier(  # sklearn 机器学习库里面自带的SGDC分类器-->随机梯度
        loss='log',  # loss function: logistic regression 逻辑回归
        penalty='elasticnet',  # L1 & L2 弹性方式、两者结合
        fit_intercept=True,  # 是否存在截距，默认存在
        n_iter=1,  #最大的迭代次数，本来等于100
        shuffle=True,  # Whether or not the training data should be shuffled after each epoch  每次迭代后是否随机打乱
        n_jobs=1,  # The number of processors to use
        class_weight=None)  # Weights associated with classes. If not given, all classes are supposed to have weight one.
    # 同样的权重

    # 管道机制使得参数集在新数据集（比如测试集）上的重复使用，管道机制实现了对全部步骤的流式化封装和管理。
    model = Pipeline(steps=[
        ('ss', StandardScaler()),  # transformer
        ('en', classifier())  # estimator
    ])

    parameters = {
        'en__alpha': [0.001, 0.01, 0.1],
        'en__l1_ratio': [0.001, 0.01, 0.1]  # 交叉验证时需要
    }

    # StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
    folder = StratifiedKFold(n_splits=3, shuffle=True)
    # Exhaustive search over specified parameter values for an estimator.
    grid_search = GridSearchCV(  # 网格搜索的方式交叉验证，选择最好的超参数
        model,
        parameters,
        cv=folder,
        n_jobs=-1,  # -1 means using all processors
        verbose=1)
    grid_search = grid_search.fit(data[predictors],
                                  data['label'])
    return grid_search

predictors = original_feature#训练模型
model = check_model(train, predictors)

# valid predict 输出概率 验证集
# y_valid_pred = model.predict_proba(valid[predictors])
# valid1 = valid.copy()
# valid1['pred_prob'] = y_valid_pred[:, 1]


# avgAUC calculation 计算auc值，只是评估用的，和模型没有直接关系
# vg = valid1.groupby(['Coupon_id'])
# aucs = []
# for i in vg:
#     tmpdf = i[1]
#     if len(tmpdf['label'].unique()) != 2: #label只有一类 则跳过
#         continue
#     fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
#     aucs.append(auc(fpr, tpr))
# print(np.average(aucs))

# test prediction for submission
y_test_pred = model.predict_proba(dftest[predictors])
dftest1 = dftest[['User_id','Coupon_id','Date_received']].copy()
dftest1['Probability'] = y_test_pred[:,1]
dftest1.to_csv('submit2.csv', index=False, header=False)

print(dftest1.head(5))