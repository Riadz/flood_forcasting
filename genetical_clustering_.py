import math
import random as rd
import numpy as np
import functools as ft
from jmetal.algorithm.singleobjective import GeneticAlgorithm

from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution

rand_seed = 99
np.random.seed(rand_seed)
rd.seed(rand_seed)

# MAX VALUE: 65000


# class OneMax(BinaryProblem):
#   def __init__(self, number_of_bits: int = 256):
#     super(OneMax, self).__init__()
#     self.number_of_bits = number_of_bits
#     self.number_of_objectives = 1
#     self.number_of_variables = 1
#     self.number_of_constraints = 0

#     self.obj_directions = [self.MINIMIZE]
#     self.obj_labels = ["Ones"]

#   def evaluate(self, solution: BinarySolution) -> BinarySolution:
#     counter_of_ones = 0
#     for bits in solution.variables[0]:
#       if bits:
#         counter_of_ones += 1

#     solution.objectives[0] = -1.0 * counter_of_ones

#     return solution

#   def create_solution(self) -> BinarySolution:
#     new_solution = BinarySolution(
#         number_of_variables=1, number_of_objectives=1)
#     new_solution.variables[0] = [True if rd.randint(
#         0, 1) == 0 else False for _ in range(self.number_of_bits)]
#     return new_solution

#   def get_name(self) -> str:
#     return "OneMax"


def main():
  DATA = np.genfromtxt(
      r'./data/Data_vietnam.csv', delimiter=','
  )[1:]

  gen_clu = GenCluster(DATA, 1)
  np.savetxt('genetical_clustering_.log', gen_clu.pop_bin, fmt='%s')

  for i in range(gen_clu.pop_size):

    print(i, gen_clu.pop[i])
    print(i, get_set(rand_seed, gen_clu.pop[i][0][0], 15784))

  # a = OneMax()
  # print(a.create_solution())

  print('end')


class GenCluster():
  def __init__(self, data, pop_size, pop_min=200):
    self.data = np.array(data)
    self.pop_size = int(pop_size)
    self.pop_min = int(pop_min)
    self.pop_count = int(data.shape[0])
    self.k_min = 2
    self.k_max = int(self.pop_count/self.pop_min)

    self.bin_size = 16
    self.bin_len = self.k_max * self.bin_size * 2

    self.pop = self.gen_population()
    self.pop_bin = []  # self.calc_pop_bin()

    print(
        '***',
        f'k ∈ [2 …  {self.k_max}] ',
        f'p = {self.pop_count}, min_p = {self.pop_min}',
        '***',
        sep='\n'
    )

  def gen_population(self):
    pop = []

    for i in range(self.pop_size):
      k = rd.randint(self.k_min, self.k_max)
      chrom = []

      for i in range(k):
        li_max = self.calc_li_max(i, k, chrom)
        li = rd.randint(self.pop_min, li_max) if i != k - 1 else li_max

        idi_max = self.calc_idi_max(li_max)
        idi = rd.randint(1, idi_max)

        chrom.append([li, idi])

      pop.append(chrom)

    return pop

  # fit
  def fit(self):
    return

  # calc
  def calc_li_max(self, i, k, ind):
    return (
        self.pop_count - ft.reduce(lambda a, b: a+b[0], ind, 0)
    ) - (
        ((k-1) - i) * self.pop_min
    )

  def calc_idi_max(self, li_max):
    return get_comb(self.pop_count, li_max)

  def calc_pop_bin(self, set=False):
    def bin_func(x): return bin(x)[2:].zfill(self.bin_size)
    bin_vec = np.vectorize(bin_func)

    pop_bin = []

    for i in range(self.pop_size):
      pop_np = np.array(self.pop[i]).flatten()
      pop_np = bin_vec(pop_np)
      pop_np = ''.join(pop_np)
      pop_np = pop_np.zfill(self.bin_len)

      pop_bin.append(pop_np)

    if set:
      self.pop_bin = pop_bin

    return pop_bin

  def calc_pop_bin_reverse(self, set=False):
    pop = []
    for i in range(self.pop_size):
      chrom = self.pop_bin[i]
      chrom = [
          chrom[j:j+self.bin_size]
          for j in range(0, self.bin_len, self.bin_size)
      ]
      chrom = [
          int(chrom[j], 2)
          for j in range(len(chrom))
      ]
      chrom = chrom[get_first_positive_index(chrom):]
      chrom = [
          chrom[j:j+2]
          for j in range(0, len(chrom), 2)
      ]

      pop.append(chrom)

    if set:
      self.pop = pop

    return pop

  def calc_fitness(self, decoded_chrom):
    k = len(decoded_chrom)
    for i in range(k):
      [li, idi] = decoded_chrom[i]

      set = get_set(rand_seed, li, idi)

      # print(f'li: {li}, idi: {idi}')
      # print(set)

    return decoded_chrom


def get_set(Ralpha=0, P=0,  ID=0):
  pool = list()
  pool.append(1)
  SOM = 0

  for j in range(1, Ralpha+1):
    cSOM = SOM
    k = 0
    while (cSOM <= ID):
      SOM = cSOM
      c = P-(pool[j-1]+(k))

      d = Ralpha-(j)
      f = get_comb(c, d)
      cSOM += f
      if(cSOM <= ID):
        k = k+1

    pool[j-1] = pool[j-1]+(k)
    if(j-1 < Ralpha-1):
      pool.append(pool[j-1]+1)

  return pool


def get_comb(n, r):
  if (r <= n):
    return (math.factorial(n) // (math.factorial(r) * math.factorial(n - r)))
  else:
    return 0


def get_first_positive_index(arr: list):
  for i in range(len(arr)):
    if arr[i] > 0:
      return i
  return -1


#
#
if __name__ == '__main__':
  main()
