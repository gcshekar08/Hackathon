import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KR


def year_month(df_date):
    #Separate date to year and month
    ty = []
    tm = []
    for i in date:
        year, month = re.findall('.{1,4}', str(i))
        ty.append(year)
        tm.append(month)
    return ty, tm

def encode(df_01, df_02):
    a = []
    b = len(df_01)
    for i in df_01:
        a.append(i)
    for i in df_02:
        a.append(i)
    le = LabelEncoder()
    a = le.fit_transform(a)
    
    df_01 = a[0:b]
    df_02 = a[b::]
    return df_01, df_02

#Read train file
df = pd.read_csv('train.csv')

date = df['Date']
m = len(date)

#From Train Separate date to year and month
df['year'], df['month'] = year_month(date)

#Read test file
df_2 = pd.read_csv('test.csv')

#Read submission file
df_3 = pd.read_csv('sample_submission.csv')

test_df = df_2['UID']

#Find test data in train data and get Gender and Age
a = []
g = []
c = 0
ag = 0
ge = 0
for i in test_df:
    c += 1
    print(c)
    df_copy = df.loc[df['UID'].isin([i])]
    train_id = df_copy['UID']
    year = df_copy['year']
    gender = df_copy['Gender']
    age = df_copy['Age']
    for l, m in zip(gender, age):
        ge = l
        ag = m 

    a.append(ag)
    g.append(ge)

df_2['Gender'] = g
df_2['Age'] = a

#Create new column with 2014 for test
c = []
for i in range(len(df_2['UID'])):
    c.append(2014)

df_2['year'] = c
#Find year with 2012 and 2013 in train data year and move age
year = df['year']
age = df['Age']
age_1 = []
for i, j in zip(year, age):
    if int(i) == 2012:
        j = j 
    if int(i) == 2013:
        j = j 
    age_1.append(j)

df['Age'] = age_1


df_3['UID'] = df_2['UID']

target = df['Event_Code']

#Lable Encode 
df['UID'], df_2['UID'] = encode(df['UID'], df_2['UID'])

#GEt dummies for train data
train = pd.concat([df.drop(['Gender', 'Event_Code', 'Date'], axis=1),
                  pd.get_dummies(df['Gender'])], axis=1)

#Get dummies for test data
test = pd.concat([df_2.drop(['Gender'], axis=1), pd.get_dummies(df_2['Gender'])], axis=1)

#Create month column with 06 in test data
c = []
for k in range(len(test['UID'])):
    c.append('06')
test['month'] = c
    
#Predict with KNeighborsClassifier
est = KR(n_neighbors=10, weights='distance', metric='manhattan', n_jobs=-1)
est.fit(train, target)
y = est.kneighbors(test, n_neighbors=10, return_distance=False)
y = np.asarray(y)
y = y.astype(np.int64)
a = []
for i in y:
    b = []
    for j in i:
        b.append(target[j])
    a.append(b)

df_a = pd.DataFrame(a, columns=['Event1', 'Event2', 'Event3', 'Event4', 'Event5', 'Event6', 'Event7', 'Event8', 'Event9', 'Event10'])
#Create submission file
df_a.to_csv('Result_5.csv')
