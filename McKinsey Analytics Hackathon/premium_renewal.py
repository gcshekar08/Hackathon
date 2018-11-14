import pandas as pd
import numpy as np
import xgboost as xgb
import math

#Train Data set
df = pd.read_csv('train_ZoGVYWq.csv')

target = df['renewal']

df = df.drop(['id', 'residence_area_type', 'sourcing_channel', 'renewal'], axis=1)

#Replace Nan with mean value
df.fillna(df.mean(), inplace=True)

#Calculate %premium paid * no.of.Premium paid * premium for test data set
X = []
b = 0
for i, j, k in zip(df['perc_premium_paid_by_cash_credit'], df['no_of_premiums_paid'], df['premium']):
    b = ((i * j) + 0.75)*k
    b = math.log(b)
    X.append(b)
    
df['premium_paid'] = X

train = df.drop(['perc_premium_paid_by_cash_credit', 'no_of_premiums_paid', 'premium'], axis=1)

train = xgb.DMatrix(train, target)
#Test Data set
df_2 = pd.read_csv('test_66516Ee.csv')
test_id = df_2['id']

df_2 = df_2.drop(['id', 'residence_area_type', 'sourcing_channel'], axis=1)

#Replace Nan with mean value
df_2.fillna(df_2.mean(), inplace=True)

#Calculate %premium paid * no.of.Premium paid * premium for test data set 
X_1 = []
b = 0
for i, j, k in zip(df_2['perc_premium_paid_by_cash_credit'], df_2['no_of_premiums_paid'], df_2['premium']):
    b = ((i * j) + 0.75)*k
    b = math.log(b)
    X_1.append(b)
    
df_2['premium_paid'] = X_1

test = df_2.drop(['perc_premium_paid_by_cash_credit', 'no_of_premiums_paid', 'premium'], axis=1)

test = xgb.DMatrix(test)
#Sample Data set
df_3 = pd.read_csv('sample_submission_sLex1ul.csv')

#Gradient Boosting Classifier
clf = xgb.train({'eta':0.1, 'booster':'gbtree'}, train, num_boost_round = 45)
y = clf.predict(test)


df_3['id'] = test_id
df_3['renewal'] = y
premium = df_2['premium']
print(min(y))

#Calculate Incentives 
a = []
b = []
c = 0
d = 0
for i, j in zip(df_3['renewal'], premium):
    d = i
    if i > 0.7:
        c = j * 0.025
        if i > 1:
            d = 1
    if i < 0.7:
        c = j * 0.05
        if i<0:
            d = 0
    a.append(c)
    b.append(d)
    
df_3['incentives'] = a
df_3['renewal'] = b

#Submission 
df_3.to_csv('result_13.csv',header=True, index=False, index_label=['id'])
