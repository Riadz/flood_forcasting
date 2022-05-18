from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
import pandas as pd
import numpy as np
# 
data_url = r'./data/Data.xlsx'
data_egypt = pd.read_excel(data_url, sheet_name='Egypt')
data_vietnam = pd.read_excel(data_url, sheet_name='Vietnam')

DATA = data_vietnam
DATA = DATA.to_numpy()

train_x, test_x, train_y, test_y = train_test_split(
  DATA[:, 1:], DATA[:, 0], test_size=0.8, random_state=1
)

print(train_x.shape, train_y.shape)

tree_classifier = tree.DecisionTreeClassifier()
tree_classifier.fit(train_x, train_y)

pred_y = tree_classifier.predict(test_x)

print("ACC:", metrics.accuracy_score(test_y, pred_y))
print("RMSE:", metrics.mean_squared_error(test_y, pred_y))
print(metrics.confusion_matrix(test_y, pred_y))
