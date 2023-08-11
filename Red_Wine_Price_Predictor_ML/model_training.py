import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score as asc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Importing Dataset
df = pd.read_csv('Red.csv')


# EDA & Preprocessing
print(df.info())

df = df.drop(['Name', 'Winery', 'NumberOfRatings'], axis= 1)
df['Price'] = df['Price'].round()

for price in df['Price'].values:
    if price <= 5.0 :
        df['Price'] = df['Price'].replace(price, '0-5')
    elif price > 5.0 and price <= 10.0 :
        df['Price'] = df['Price'].replace(price, '5-10')
    elif price > 10.0 and price <= 25.0 :
        df['Price'] = df['Price'].replace(price, '10-25')
    elif price > 25.0 and price <= 50.0 :
        df['Price'] = df['Price'].replace(price, '25-50')
    elif price > 50.0 and price <= 100.0 :
        df['Price'] = df['Price'].replace(price, '50-100')
    elif price > 100.0 :
        df['Price'] = df['Price'].replace(price, '100+')
print(df['Price'])

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    df[col] = df[col].astype(float)


# Training & Test Sets Split
features = df.drop('Price', axis= 1)
target = df['Price']

X_train, X_test, Y_train, Y_test = train_test_split(features, target,
                                                    test_size= 0.25,
                                                    shuffle= True,
                                                    random_state= 42)


# Model Training & Voting
models = [XGBClassifier(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          GradientBoostingClassifier(),
          VotingClassifier([('rfc', RandomForestClassifier()),
                           ('gbc', GradientBoostingClassifier()),
                           ('dtc', DecisionTreeClassifier()),
                           ('xgbc', XGBClassifier())
                           ])]

for m in models:
    print(m)

    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(f'Train Accuracy is : {asc(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy is : {asc(Y_test, pred_test)}\n')
