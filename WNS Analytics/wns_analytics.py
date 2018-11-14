import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

df = pd.read_csv('train_LZdllcl.csv')

print( df.isnull().sum(axis = 0))


df['previous_year_rating'].fillna(5, inplace=True)

print( df.isnull().sum(axis = 0))

target = df['is_promoted']


train = pd.concat([df.drop(['department', 'employee_id', 'is_promoted','recruitment_channel'], axis=1), 
                   pd.get_dummies(df['department'])],axis=1)

train = pd.concat([train.drop(['region'], axis=1), 
                   pd.get_dummies(train['region'])],axis=1)

train = pd.concat([train.drop(['education'], axis=1), 
                   pd.get_dummies(train['education'])],axis=1)

train = pd.concat([train.drop(['gender'], axis=1), 
                   pd.get_dummies(train['gender'])],axis=1)

#train = pd.concat([train.drop(['recruitment_channel'], axis=1), 
#                   pd.get_dummies(train['recruitment_channel'])],axis=1)

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.3, random_state=45)

clf = RandomForestClassifier(n_estimators=15,criterion='entropy',max_features=None,random_state=45)
#clf = RandomForestClassifier(random_state=45)
#clf = XGBClassifier()
clf.fit(X_train, y_train)

y = clf.predict(X_test)

print(f1_score(y_test, y, average='macro'))
