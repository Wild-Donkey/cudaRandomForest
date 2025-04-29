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

OverallS = datetime.datetime.now().timestamp()

for i in range(data.shape[1]):
  train_label = train_data.iloc[:, i]
  test_label = test_data.iloc[:, i]
  train_feature = train_data.drop(columns=[train_data.columns[i]])
  test_feature = test_data.drop(columns=[test_data.columns[i]])

  print("train_feature shape", train_feature.shape)
  print("train_label shape", train_label.shape)

  if(data.columns[i] in data_encoders.keys()):
    RF = TGBMClassifier()
  else :
    RF = TGBMRegressor()

  Start = datetime.datetime.now().timestamp()
  print("feat Shape", train_feature.shape)
  print("label Shape", train_label.shape)
  RF.fit(train_feature, train_label)
  End = datetime.datetime.now().timestamp()
  print("Fit:", train_size, " Time:" + str(End - Start) + "s")

  Start = datetime.datetime.now().timestamp()
  test_predict = RF.predict(test_feature)
  End = datetime.datetime.now().timestamp()
  print("Predict:", test_size, " Time:" + str(End - Start) + "s")

  Loss = np.mean(np.square(test_predict - test_label))

  print("Loss", Loss)
  if(data.columns[i] in data_encoders.keys()):
    test_predict = np.round(test_predict).astype(int)
    test_predict = data_encoders[data.columns[i]].inverse_transform(test_predict)
  df[train_data.columns[i]] = test_predict
  del RF

OverallEnd = datetime.datetime.now().timestamp()
print("Overall Time:" + str(OverallEnd - OverallS) + "s")
df.to_csv('../data/adult_predict.csv', index = False)