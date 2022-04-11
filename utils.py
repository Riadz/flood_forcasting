import torch
import pandas as pd


def toTorchTensers(*args, device=torch.device('cpu')):
  val = []
  for arg in args:
    val.append(torch.tensor(
        arg,
        dtype=torch.float,
        # requires_grad=True,
        device=device
    ))

  return val


def NormalizeData(data) -> pd.DataFrame:
  def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

  for col in data.columns:
    data[col] = min_max_scaling(data[col])
  return data
