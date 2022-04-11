import math
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer

base_url = r'.'


class AutoEncoder(nn.Module):
  def __init__(self, layers=[8, 6]):
    super().__init__()

    in_net = 12
    out_net = 3

    hidden_net = [nn.ReLU()]
    for i in range(len(layers)-1):
      hidden_net += [
          nn.Linear(layers[i], layers[i+1]),
          nn.ReLU(),
      ]

    layers_reversed = layers[:]
    layers_reversed.reverse()
    hidden_net_reverse = [nn.ReLU()]
    for i in range(len(layers_reversed)-1):
      hidden_net_reverse += [
          nn.Linear(layers_reversed[i], layers_reversed[i+1]),
          nn.ReLU(),
      ]

    self.encoder = nn.Sequential(
        nn.Linear(in_net, layers[0]),
        *hidden_net,
        nn.Linear(layers[-1], out_net),
    )
    self.decoder = nn.Sequential(
        nn.Linear(out_net, layers[-1]),
        *hidden_net_reverse,
        nn.Linear(layers[0], in_net),
        nn.Sigmoid(),
    )

    self.current_loss = None
    self.opti = None
    self.crit = nn.MSELoss()
    self.init()

  def init(self):
    self.opti = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def fit(self, data_x, data_y, epoches=800, batch_size=200, progress=True):
    items_len = data_x.shape[0]
    iterations = range(math.ceil(items_len/batch_size))
    start_time = timer()

    for epoche in range(epoches):
      for i in iterations:
        start = i*batch_size
        end = min((i+1)*batch_size, items_len)

        recon = self(data_x[start:end])
        loss = self.crit(recon, data_y[start:end])

        self.opti.zero_grad()
        loss.backward()
        self.opti.step()

      if progress:
        print(f'epoche: {epoche}, loss: {loss.item():.8f}')

    #
    exec_time = f"{(timer() - start_time):.1f}"
    print(
        f'✅ training ended ,final loss: {loss.item():.8f}, time: {exec_time}s'
    )
    self.current_loss = loss.item()

  def save(self, path=None):
    if path is None:
      from datetime import datetime
      current_datetime = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
      path = rf'{base_url}/models/auto_model_{current_datetime}.pt'

    torch.save(
        self.state_dict(),
        path
    )
    print(f'✅ model saved as "auto_model_{current_datetime}.pt"')
