from thundergbm import TGBMClassifier, TGBMRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import datetime

data = pd.read_csv('../data/adult.csv')

data_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
  le = LabelEncoder()
  data[column] = le.fit_transform(data[column])
  data_encoders[column] = le

train_size = int(data.shape[0] * 0.8)
test_size = data.shape[0] - train_size

train_data = data.iloc[0:train_size]
test_data = data.drop(train_data.index)

df = pd.DataFrame()

for i in range(data.shape[1]):
  test_label = test_data.iloc[:, i]
  df[train_data.columns[i]] = test_label

print(df.columns.tolist())
df.drop(columns=df.columns.tolist()[0], inplace=True)
df.to_csv('../data/adult_label.csv', index = False)
