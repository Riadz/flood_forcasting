import math
import decimal
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from timeit import default_timer as timer

base_url = r'.'

#


class AutoEncoder(nn.Module):
  def __init__(self, layers=[8, 6], input=12):
    super().__init__()

    in_net = input
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

  def fit__(self, data_x, data_y, epoches=100, progress=True):
    start_time = timer()

    for epoche in range(epoches):
      epoche_start_time = timer()

      for i in range(len(data_x)):
        recon = self(data_x[i])
        loss = self.crit(recon, data_y[i])

        self.opti.zero_grad()
        loss.backward()
        self.opti.step()

      epoche_exec_time = f"{(timer() - epoche_start_time):.1f}"
      if progress:
        print(
            f'epoche: {epoche}, loss: {loss.item():.8f}, execution time: {epoche_exec_time}s'
        )

    exec_time = f"{(timer() - start_time):.1f}"
    print(
        f'✅ training ended ,final loss: {loss.item():.8f}, time: {exec_time}s'
    )

  def save(self, name='auto_model'):
    current_datetime = datetime.now().strftime('%d-%m_%H-%M-%S')
    path = rf'{base_url}/models/{name}_{current_datetime}.pt'

    torch.save(
        self.state_dict(),
        path
    )
    print(f'✅ model saved as "{name}_{current_datetime}.pt"')

#


class ForcastModel(nn.Module):
  def __init__(self, input=12):
    super().__init__()

    self.pipeline = nn.Sequential(
        nn.Linear(input, 48),
        nn.ReLU(),
        nn.Linear(48, 48),
        nn.ReLU(),
        nn.Linear(48, 24),
        nn.ReLU(),
        nn.Linear(24, 24),
        nn.ReLU(),
        nn.Linear(24, 24),
        nn.ReLU(),
        nn.Linear(24, 1),
        nn.Sigmoid()
    )

    self.current_loss = None
    self.opti = None
    self.crit = nn.MSELoss()
    self.init()

  def init(self):
    self.opti = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

  def forward(self, x):
    return self.pipeline(x)

  def fit(self, data_x, data_y, epoches=100, progress=True):
    start_time = timer()

    for epoche in range(epoches):
      epoche_start_time = timer()

      for i in range(len(data_x)):
        recon = self(data_x[i])
        loss = self.crit(recon, data_y[i])

        self.opti.zero_grad()
        loss.backward()
        self.opti.step()

      epoche_exec_time = f"{(timer() - epoche_start_time):.1f}"
      if progress:
        print(
            f'epoche: {epoche}, loss: {loss.item():.8f}, execution time: {epoche_exec_time}s'
        )

    exec_time = f"{(timer() - start_time):.1f}"
    print(
        f'✅ training ended ,final loss: {loss.item():.8f}, time: {exec_time}s'
    )

  def fit_batch(self, data_x, data_y, epoches=800, batch_size=200, progress=True):
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

  def save(self, name="forcast_model"):
    current_datetime = datetime.now().strftime("%d-%m_%H-%M-%S")
    path = rf'{base_url}/models/{name}_{current_datetime}.pt'

    torch.save(
        self.state_dict(),
        path
    )
    print(f'✅ model saved as "{name}_{current_datetime}.pt"')

  def test(self, data_x, data_y, bar=0.7):
    # test
    acc_array = []
    for i in range(len(data_x)):
      recon = self(data_x[i])
      loss_ = self.crit(recon, data_y[i])
      recon_ = 1 if recon[0].item() > bar else 0
      acc_array.append(1 if data_y[i] == recon_ else 0)

    print(f'loss: {loss_:.4f}, acc: {sum(acc_array)/len(data_y):.4f}')


class ForcastModelMulti(nn.Module):
  def __init__(self, input=12, output=1):
    super().__init__()

    self.pipeline = nn.Sequential(
        nn.Linear(input, 48),
        nn.ReLU(),
        nn.Linear(48, 48),
        nn.ReLU(),
        nn.Linear(48, 24),
        nn.ReLU(),
        nn.Linear(24, 24),
        nn.ReLU(),
        nn.Linear(24, 24),
        nn.ReLU(),
        nn.Linear(24, output),
        nn.Softmax(),
    )

    self.current_loss = None
    self.opti = None
    self.crit = nn.MSELoss()
    self.init()

  def init(self):
    self.opti = optim.SGD(self.parameters(), lr=1e-1)

  def forward(self, x):
    return self.pipeline(x)

  def fit(self, data_x, data_y, epoches=100, progress=True):
    start_time = timer()
    print(
        f'⚙ training ...'
    )

    for epoche in range(epoches):
      epoche_start_time = timer()

      for i in range(len(data_x)):
        Y = torch.zeros(
            int(torch.max(data_y).item())
        )
        Y[int(data_y[i].item() - 1)] = 1

        recon = self(data_x[i])
        loss = self.crit(recon, Y)

        self.opti.zero_grad()
        loss.backward()
        self.opti.step()

      epoche_exec_time = f"{(timer() - epoche_start_time):.1f}"
      if progress:
        print(
            f'epoche: {epoche}, loss: {loss.item():.12f}, execution time: {epoche_exec_time}s'
        )

    exec_time = f"{(timer() - start_time):.1f}"
    print(
        f'✅ training ended ,final loss: {loss.item():.8f}, time: {exec_time}s'
    )

  def fit_batch(self, data_x, data_y, epoches=800, batch_size=200, progress=True):
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

  def save(self, name="forcast_model"):
    current_datetime = datetime.now().strftime("%d-%m_%H-%M-%S")
    path = rf'{base_url}/models/{name}_{current_datetime}.pt'

    torch.save(
        self.state_dict(),
        path
    )
    print(f'✅ model saved as "{name}_{current_datetime}.pt"')

  def test(self, data_x, data_y, bar=0.7):
    # test
    acc_array = []
    for i in range(len(data_x)):
      Y = torch.zeros(
          int(torch.max(data_y).item())
      )
      Y[int(data_y[i].item() - 1)] = 1

      recon = self(data_x[i])
      loss = self.crit(recon, Y)
      acc_array.append(
          1 if torch.argmax(Y) == torch.argmax(recon) else 0
      )

    print(f'loss: {loss:.4f}, acc: {sum(acc_array)/len(data_y):.4f}')
