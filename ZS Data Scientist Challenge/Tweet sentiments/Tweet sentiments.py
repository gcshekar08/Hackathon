import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as GBR
from sklearn.preprocessing import LabelEncoder
import re
import numpy as np


def created(from_val, to_val):
    #Move datetime from tweet loaction to tweet created
    loc = []
    cre = []
    for i, j in zip(from_val, to_val):
        try:
            if re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} -\d{4}', i) is None:
                pass
            else:
                i = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} -\d{4}', i)
                i = i.group()
                temp = j
                j = i
                i = temp
        except TypeError:
            pass
        except AttributeError:
            pass
        loc.append(i)
        cre.append(j)
    return loc, cre

def time_zone(from_val, to_val):
    #Move timezone from tweet location to user timezone
    loc = []
    timezone = []
    for i, j in zip(from_val, to_val):
        try:
            if re.search(r'Time', j) is None:
                temp = j
                j = i
                i = temp
        except TypeError:
            pass
        loc.append(i)
        timezone.append(j)
    return loc, timezone

def only_date(date_time):
    # Extracting date from datetime
    a = []
    z = 0
    for i in date_time:
        z = re.sub('\d{2}:\d{2}:\d{2} -\d{4}','', str(i))
        a.append(z)
    return a

#Read train file
df = pd.read_csv('train.csv', usecols=['tweet_id', 'airline_sentiment', 'airline_sentiment_confidence',
                                       'negativereason_confidence', 'retweet_count', 'tweet_created',
                                       'tweet_location', 'user_timezone', 'negativereason', 'airline'])
#Read test file
df_2 = pd.read_csv('test.csv', usecols=['tweet_id', 'airline_sentiment_confidence',
                                       'negativereason_confidence', 'retweet_count', 'tweet_created',
                                       'tweet_location', 'user_timezone', 'negativereason', 'airline'])
#Train file tweet location
tweet_location = df['tweet_location']
#Train file tweet created time and date
tweet_created = df['tweet_created']
#Train file tweet timezone
user_timezone = df['user_timezone']

#Train Move datetime from tweet loaction to tweet created
df['tweet_location'], df['tweet_created'] = created(tweet_location, tweet_created)
#Train Move datetime from user timezone to tweet created
df['user_timezone'], df['tweet_created'] = created(user_timezone, tweet_created)
#creat dummie columns
df_2['-1'] = np.zeros(len(df_2['tweet_id']))
df_2['&lt'] = np.zeros(len(df_2['tweet_id']))
df_2['606&gt'] = np.zeros(len(df_2['tweet_id']))
df_2['-2'] = np.zeros(len(df_2['tweet_id']))

#Test file tweet location
tweet_location = df_2['tweet_location']
#Test file tweet created time and date
tweet_created = df_2['tweet_created']
#Test file tweet timezone
user_timezone = df_2['user_timezone']

#Test Move datetime from tweet loaction to tweet created
df_2['tweet_location'], df_2['tweet_created'] = created(tweet_location, tweet_created)
#Test Move datetime from user timezone to tweet created
df_2['user_timezone'], df_2['tweet_created'] = created(user_timezone, tweet_created)


tweet_created = df['tweet_created']
#From train tweet created extract only date
df['tweet_created'] = only_date(tweet_created)

tweet_created = df_2['tweet_created']
#From test tweet created extract only date
df_2['tweet_created'] = only_date(tweet_created)

tweet_location = df['tweet_location']
user_timezone = df['user_timezone']
#From train Move timezone from tweet location to user timezone
df['tweet_location'], df['user_timezone'] = time_zone(tweet_location, user_timezone)

tweet_location_2 = df_2['tweet_location']
user_timezone_2 = df_2['user_timezone']
#From test Move timezone from tweet location to user timezone
df_2['tweet_location'], df_2['user_timezone'] = time_zone(tweet_location_2, user_timezone_2)

#Create submission Dataframe
df_3 = pd.DataFrame(df_2['tweet_id'])

#Fill NaN with -1
df.fillna(-1, inplace=True)
df_2.fillna(-1, inplace=True)
#Lable Encode target
le = LabelEncoder()
target = le.fit_transform(df['airline_sentiment'])



df = pd.concat([df.drop(['airline', 'negativereason', 'tweet_location', 'user_timezone', 'airline_sentiment', 'tweet_id'], axis=1),
                   pd.get_dummies(df['airline'])], axis=1)

train = pd.concat([df.drop(['tweet_created'], axis=1), pd.get_dummies(df['tweet_created'])], axis=1)



df_2 = pd.concat([df_2.drop(['airline', 'negativereason', 'tweet_location', 'user_timezone', 'tweet_id'], axis=1),
                   pd.get_dummies(df_2['airline'])], axis=1)

test = pd.concat([df_2.drop(['tweet_created'], axis=1), pd.get_dummies(df_2['tweet_created'])], axis=1)

#GradientBoostingClassifier
est = GBR(n_estimators=100, learning_rate=0.1)
est.fit(train, target)
y = est.predict(test)
y = y.astype(np.int64)
y = np.round(y)
#Inverse Transform
y_result = le.inverse_transform(y)

df_3['airline_sentiment'] = y_result

df_3.to_csv('submission_2.csv', index=True, header=True, index_label='tweet_id')