import torch
import pandas as pd
import numpy as np
from libs.tsne import tsne
from utils import NormalizeData

# data_url = r'./data/Data.xlsx'
# data_egypt = pd.read_excel(data_url, sheet_name='Egypt')
# data_vietnam = pd.read_excel(data_url, sheet_name='Vietnam')

# DATA = data_vietnam
# DATA = NormalizeData(DATA)
# # DATA = DATA[DATA.Flood == 1]
# DATA = DATA.to_numpy()

DATA = np.genfromtxt(
    r'./data/other_banknote_authentication.csv', delimiter=','
)

X = DATA[:, :-1]
X = torch.Tensor(X)

TSNE_EMB = tsne(X, 3, 12, 20.0)
TSNE_EMB = TSNE_EMB.cpu().numpy()

np.savetxt("./tsne/tsne__.csv", TSNE_EMB, delimiter=",")
