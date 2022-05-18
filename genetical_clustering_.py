import random as rd
import pandas as pd
import numpy as np
import functools as ft
from typing import Tuple
from nptyping import NDArray

rand_seed = 2
np.random.seed(rand_seed)
rd.seed(rand_seed)


def main():
  DATA = np.genfromtxt(
      r'./data/Data_vietnam.csv', delimiter=','
  )[1:]

  gen_clu = GenCluster(DATA)

  print('end')


class GenCluster():
  def __init__(self, data, pop_min=200):
    self.data = np.array(data)
    self.pop_min = int(pop_min)
    self.pop_count = int(data.shape[0])
    self.k_min = 2
    self.k_max = int(self.pop_count/self.pop_min)
    self.k = rd.randint(self.k_min, self.k_max)

    self.pop = self.gen_population()

    print(
        '***\n',
        f'k = {self.k}, ∈ [2 ... {self.k_max}] \n',
        f'p = {self.pop_count}, min_p = {self.pop_min}',
    )

  def gen_population(self):
    pop = []
    for i in range(self.k):
      li_max = self.calc_li_max(i, pop)
      li = rd.randint(self.pop_min, li_max) if i != self.k - 1 else li_max

      idi_max = self.calc_idi_max(li)
      idi = rd.randint(1, idi_max)

      pop.append([li, idi])

    return pop

  # fit
  def fit(self):
    return

  # calc
  def calc_li_max(self, i, pop):
    return (
        self.pop_count - ft.reduce(lambda a, b: a+b[0], pop, 0)
    ) - (
        ((self.k-1) - i) * self.pop_min
    )

  def calc_idi_max(self, li_max):
    return int(self.pop_count / li_max)

  def calc_fitness(self, individual):
    return


#
#
if __name__ == '__main__':
  main()
