# Simpler auto model
class AutoEncoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.opti = None
    self.crit = nn.MSELoss()
    self.encoder = nn.Sequential(
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 6),
        nn.ReLU(),
        nn.Linear(6, 3),
        # nn.Sigmoid(),
    )
    self.decoder = nn.Sequential(
        nn.Linear(3, 6),
        nn.ReLU(),
        nn.Linear(6, 8),
        nn.ReLU(),
        nn.Linear(8, 12),
        nn.Sigmoid(),
    )

    self.init()

  def init(self):
    self.opti = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def fit(self, data_x, data_y, epoches=100):

    for epoche in range(epoches):
      epoche_start_time = timer()

      for i in range(len(data_x)):
        recon = self(data_x[i])
        loss = self.crit(recon, data_y[i])

        self.opti.zero_grad()
        loss.backward()
        self.opti.step()

      epoche_exec_time = f"{(timer() - epoche_start_time):.1f}"
      print(
          f'epoche: {epoche}, loss: {loss.item():.8f}, execution time: {epoche_exec_time}s'
      )


# test one sample
t = test_x[2]
recon = model(t)
loss = model.crit(recon, t)
print(f'loss {loss.item(): .4f}')
for i in range(len(t)):
  print(f'expected: {t[i]: .4f} --got-> {recon[i]: .4f}')

#
