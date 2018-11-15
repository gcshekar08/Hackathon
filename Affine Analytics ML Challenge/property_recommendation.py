import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.classification import f1_score

def combine(df_to, df_from, df_combined):
    for i in df_to['id_accs']:
        df_combined = df_combined.append(df_from.loc[df_from['id_accs'] == i])
    return df_combined

def con_bool(column, df):    
    for i in col_bool:
        df[i] = df[i].replace(True, 1)
        df[i] = df[i].replace(False, 0)
    return df


df = pd.read_csv('train_3.csv')

df_1 = pd.read_csv('Accounts.csv')

column = list(df_1.columns.values)
df_com = pd.DataFrame(columns=column)

df_2 = pd.read_csv('Test_Data.csv')

test = combine(df_2, df_1, df_com)

submission = pd.DataFrame(columns=['id_accs', 'id_prop'])

submission['id_accs'] = test['id_accs']

a = []
b = 0
for i in range(0,len(test)):
    b = 2018
    a.append(b)
    
test['year'] = a

print(test.shape)

print(df.shape)

col_bool = ['buyer_book', 'servicing_contract', 'cmbs', 'consultant', 'correspondent', 'foreign', 
            'master_servicer', 'lender_book', 'loan_sales_book', 'loan_servicing']


train =  con_bool(col_bool, df)  

test =  con_bool(col_bool, test)

print(df.describe())

print(df.isnull().sum())

target = df['id_props']   

df.drop(['buyer_book', 'servicing_contract', 'cmbs', 'consultant', 'correspondent', 
         'foreign', 'master_servicer', 'lender_book', 'loan_sales_book', 
         'loan_servicing', 'investor_type', 'id_props', 'id_accs'], axis=1, inplace=True) 

train = df

test.drop(['buyer_book', 'servicing_contract', 'cmbs', 'consultant', 'correspondent', 
         'foreign', 'master_servicer', 'lender_book', 'loan_sales_book', 
         'loan_servicing', 'investor_type', 'id_accs'], axis=1, inplace=True)

test = test

print(train.describe()) 
print(test)

column = train.columns
con_columns = ['active_deals', 'activity_count', 'num_deals_as_client', 
               'num_deals_as_investor', 'number_of_properties', 'number_of_related_deals',
               'number_of_related_properties', 'number_of_won_deals_as_client']


for i in con_columns:
    year = (2018 -  train['year']) + 1
    train[i] = (train[i] + 1 ) * year
    test[i] = (test[i] + 1 ) * year

train['active_deals_2'] = train['active_deals']**2
train['activity_count_2'] = train['activity_count']**2
train['active_deals_count'] = train['active_deals'] * train['activity_count']


train['num_deals_as_client_2'] = train['num_deals_as_client']**2
train['num_deals_as_investor_2'] = train['num_deals_as_investor']**2
train['num_deals_client_investor'] = train['num_deals_as_client'] * train['num_deals_as_investor']


train['num_of_properties_deals'] = train['number_of_properties'] * train['number_of_related_deals']
train['num_of_properties_related'] = train['number_of_properties'] * train['number_of_related_properties']
train['num_of_related_deals_properties'] = train['number_of_related_deals'] * train['number_of_related_properties']
train['num_of_prperties_related_deals'] = train['number_of_properties'] * train['number_of_related_deals'] * train['number_of_related_properties']

test['active_deals_2'] = test['active_deals']**2
test['activity_count_2'] = test['activity_count']**2
test['active_deals_count'] = test['active_deals'] * test['activity_count']


test['num_deals_as_client_2'] = test['num_deals_as_client']**2
test['num_deals_as_investor_2'] = test['num_deals_as_investor']**2
test['num_deals_client_investor'] = test['num_deals_as_client'] * test['num_deals_as_investor']


test['num_of_properties_deals'] = test['number_of_properties'] * test['number_of_related_deals']
test['num_of_properties_related'] = test['number_of_properties'] * test['number_of_related_properties']
test['num_of_related_deals_properties'] = test['number_of_related_deals'] * test['number_of_related_properties']
test['num_of_prperties_related_deals'] = test['number_of_properties'] * test['number_of_related_deals'] * test['number_of_related_properties']

enc = LabelEncoder()    
target = enc.fit_transform(target.values)
    
model = RandomForestClassifier(n_estimators=10, criterion='entropy', max_features=None, random_state=7)
model.fit(train, target)
y = model.predict(test)
print(np.round(y))
a = []
b = 0
for i in np.round(y):
    b = int(i)
    a.append(b)
y = a    
y = enc.inverse_transform(y)

submission['id_prop'] = y

submission.to_csv('result_1.csv', index=False)
