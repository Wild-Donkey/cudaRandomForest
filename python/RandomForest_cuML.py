from joblib import Parallel, delayed
from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import datetime

def process_column(i, train_data, test_data, data_encoders, train_size, test_size):
  train_label = train_data.iloc[:, i]
  test_label = test_data.iloc[:, i]
  train_feature = train_data.drop(columns=[train_data.columns[i]])
  test_feature = test_data.drop(columns=[test_data.columns[i]])

  print(f"Processing column {train_data.columns[i]}")
  print("train_feature shape", train_feature.shape)
  print("train_label shape", train_label.shape)

  if train_data.columns[i] in data_encoders.keys():
      model = RandomForestClassifier()
  else:
      model = RandomForestRegressor()

  # 训练模型
  start = datetime.datetime.now().timestamp()
  print("feat Shape", train_feature.shape)
  print("label Shape", train_label.shape)
  model.fit(train_feature, train_label)
  end = datetime.datetime.now().timestamp()
  print(f"Fit: {train_size}, Time: {end - start}s")

  # 预测
  start = datetime.datetime.now().timestamp()
  test_predict = model.predict(test_feature)
  end = datetime.datetime.now().timestamp()
  print(f"Predict: {test_size}, Time: {end - start}s")

  # 计算损失
  loss = np.mean(np.square(test_predict - test_label))
  print("Loss", loss)

  # 处理分类结果
  if train_data.columns[i] in data_encoders.keys():
      test_predict = np.round(test_predict).astype(int)
      test_predict = data_encoders[train_data.columns[i]].inverse_transform(test_predict)
  
  del model
  return train_data.columns[i], test_predict


# 主程序
if __name__ == '__main__':
  # 假设这些变量已经定义
  # data, train_data, test_data, data_encoders, train_size, test_size, df
  data = pd.read_csv('../data/adult.csv')
  data_encoders = {}
  for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    data_encoders[column] = le
  # 并行处理所有列
  train_size = int(data.shape[0] * 0.8)
  test_size = data.shape[0] - train_size

  train_data = data.iloc[0:train_size]
  test_data = data.drop(train_data.index)

  df = pd.DataFrame()
  OverallS = datetime.datetime.now().timestamp()
  results = Parallel(n_jobs=-1, verbose=10)(
      delayed(process_column)(i, train_data, test_data, data_encoders, train_size, test_size)
      for i in range(train_data.shape[1])
  )
  
  # 将结果存入df
  for col_name, pred in results:
      df[col_name] = pred

  OverallEnd = datetime.datetime.now().timestamp()
  print("Overall Time:" + str(OverallEnd - OverallS) + "s")
  df.to_csv('../data/adult_predict.csv', index = False)
