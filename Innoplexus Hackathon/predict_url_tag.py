import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.classification import f1_score
from sklearn.neighbors import KNeighborsClassifier

def meta_data(df_from, df_to):
    meta = []
    a = []
    li = []
    div = []
    c = 0
    b = 0
    e = 0
    f = 0
    g = 0
    for i in df_to['Webpage_id']:
        b = df_from.loc[df_from['Webpage_id'] == i]
        for j in b['Html']:
            html = j
            soup = bs(html, "lxml")
            c = len(soup.find_all('meta'))
            e = len(soup.find_all('a'))
            f = len(soup.find_all('li'))
            g = len(soup.find_all('div'))
            meta.append(c)
            a.append(e)
            li.append(f)
            div.append(g)
            b = 0
    return meta, a, li, div

#Reading Html Data file
df_html = pd.read_csv('html_data.csv')

#Reading Train data
df_train = pd.read_csv('train.csv')

#Getting extra data from Html data for Train    
df_train['meta_len'], df_train['a_len'], df_train['li_len'], df_train['div_len'] = meta_data(df_html, df_train)

#reading Test data
df_test = pd.read_csv('test_nvPHrOx.csv')

submission = pd.read_csv('submission.csv')

submission['Webpage_id'] = df_test['Webpage_id']
    
#Getting extra data from Html data for Test
df_test['meta_len'], df_test['a_len'], df_test['li_len'], df_test['div_len'] = meta_data(df_html, df_test)


#Filling NaN with -1
df_train.fillna(-1, inplace=True)

df_test.fillna(-1, inplace=True)

target = df_train['Tag']

length = len(df_test)

#Encoding Test Data
en = LabelEncoder()
target = en.fit_transform(target)

# Dropping columns['Webpage_id', 'Tag', 'Domain', 'Url']
train = df_train.drop(['Webpage_id', 'Tag', 'Domain', 'Url'], axis=1)

test = df_test.drop(['Webpage_id', 'Domain', 'Url'], axis=1)


#Taking log on selected columns to Normalize

column = ['meta_len', 'a_len', 'li_len', 'div_len']

for i in column:
    train[i] = np.log1p(train[i] + 1)
    
for i in column:
    test[i] = np.log1p(test[i] + 1)

# Train and Predict Using KNN
#model = LinearSVC()
model = KNeighborsClassifier(n_neighbors=1, weights='distance')
model.fit(train, target)
y = model.predict(test)
y = en.inverse_transform(y)

#Creating Submission
submission['Tag'] = y
print('completed Prediction')
submission.to_csv('result_3.csv', index=False, header=True)