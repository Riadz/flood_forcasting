import math
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from libs.tsne import tsne


def NormalizeData(data):
  def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

  for col in data.columns:
    data[col] = min_max_scaling(data[col])
  return data


data_url = r'./data/Data.xlsx'
data_egypt = pd.read_excel(data_url, sheet_name='Egypt')
data_vietnam = pd.read_excel(data_url, sheet_name='Vietnam')

DATA = data_vietnam
DATA = NormalizeData(DATA)
DATA = DATA[DATA.Flood == 1]
DATA = DATA.to_numpy()
print(DATA.shape)

labels = DATA[:, :1]
labels = np.array(
    list(map(lambda x: 'red' if x == 1 else 'green', labels))
)

X = DATA[:, 1:]
X = torch.Tensor(X)

TSNE_EMB = tsne(X, 3, 12, 20.0)
TSNE_EMB = TSNE_EMB.cpu().numpy()


np.savetxt(r'./tsne/data_vietnam_flood_1.csv', TSNE_EMB, delimiter=",")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('PC-1')
# ax.set_ylabel('PC-2')
# ax.set_zlabel('PC-3')

# ax.scatter(
#     TSNE_EMB[:, 0], TSNE_EMB[:, 1], TSNE_EMB[:, 2],
#     s=15, c=labels, marker='o'
# )
# plt.show()
