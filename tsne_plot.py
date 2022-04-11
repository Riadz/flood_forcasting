import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 
data_url = r'./data/Data.xlsx'
data_egypt = pd.read_excel(data_url, sheet_name='Egypt')
data_vietnam = pd.read_excel(data_url, sheet_name='Vietnam')

# Egypt all
DATA = data_egypt
DATA = DATA.to_numpy()

labels = DATA[:, :1]
labels = np.array(
    list(map(lambda x: 'red' if x == 1 else 'green', labels))
)
TSNE_EMB = np.genfromtxt(r'./tsne/tsne_egypt.csv', delimiter=',')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('PC-1')
ax.set_ylabel('PC-2')
ax.set_zlabel('PC-3')

ax.scatter(
    TSNE_EMB[:, 0], TSNE_EMB[:, 1], TSNE_EMB[:, 2],
    s=15, c=labels, marker='o'
)
plt.show()

input()

# Egypt flood = 1
TSNE_EMB = np.genfromtxt(r'./tsne/tsne_egypt_flood_1.csv', delimiter=',')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('PC-1')
ax.set_ylabel('PC-2')
ax.set_zlabel('PC-3')

ax.scatter(
    TSNE_EMB[:, 0], TSNE_EMB[:, 1], TSNE_EMB[:, 2],
    s=15, marker='o'
)
plt.show()