import math
import random
import pandas as pd
import numpy as np
from typing import Tuple
from nptyping import NDArray

np.random.seed(1)


def main():
  DATA = np.genfromtxt(
      r'./data/Data_vietnam.csv', delimiter=','
  )[1:]

  gen_clu = GenCluster(DATA)

  print(gen_clu.population)

  print('end')


class GenCluster():
  def __init__(self, data, population_count=4):
    self.data = np.array(data)
    self.data_size = data.shape[0]
    self.population_count = int(population_count)
    self.population = self.gen_population()

  def gen_population(self):
    r = np.arange(self.data_size)
    np.random.shuffle(r)
    return np.array_split(r, self.population_count)

  def fit(self):
    fitness = []
    for i in range(self.population_count):
      fitness.append(
          self.calc_fitness(self.population[i])
      )

  def calc_fitness(self, individual):
    # data = self.data[:]
    # for data in range(self.data_size):
    pass


#
if __name__ == '__main__':
  main()
