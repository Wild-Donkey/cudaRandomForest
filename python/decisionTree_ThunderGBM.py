from thundergbm import TGBMClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import datetime

data = pd.read_csv('../data/adult.csv')

features = data.iloc[:, 0:14]
labels = data.iloc[:, 14]

label_encoders = {}
for column in features.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column])
    label_encoders[column] = le

if labels.dtype == 'object':
    le_label = LabelEncoder()
    labels = le_label.fit_transform(labels)

train_size = int(features.shape[0] * 0.8)
test_size = features.shape[0] - train_size

train_feature = features[0:train_size]
train_label = labels[0:train_size]
test_feature = features[train_size : features.shape[0]]
test_label = labels[train_size : features.shape[0]]

Clf = TGBMClassifier()

Start = datetime.datetime.now().timestamp()
Clf.fit(train_feature, train_label.reshape(-1))
End = datetime.datetime.now().timestamp()
print("Fit:", train_size, " Time:" + str(End - Start) + "s")

Start = datetime.datetime.now().timestamp()
test_predict = Clf.predict(test_feature)
End = datetime.datetime.now().timestamp()
print("Predict:", test_size, " Time:" + str(End - Start) + "s")

Loss = np.sum(np.square(test_predict - test_label)) / test_feature.shape[0]

print("Loss", Loss)