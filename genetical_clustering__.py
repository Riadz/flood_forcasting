import math
import random as rd
import numpy as np
import functools as ft

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from jmetal.operator import SPXCrossover, BitFlipMutation, BestSolutionSelection
from jmetal.util.termination_criterion import StoppingByEvaluations

rand_seed = 88
np.random.seed(rand_seed)
rd.seed(rand_seed)


def main():
  DATA = np.genfromtxt(
      r'./data/Data_vietnam.csv', delimiter=','
  )[1:]

  gen_clu = GenClust(DATA)

  gen_alg = GeneticAlgorithm(
      problem=gen_clu,
      population_size=100,
      offspring_population_size=100,
      mutation=BitFlipMutation(0.8),
      crossover=SPXCrossover(0.8),
      termination_criterion=StoppingByEvaluations(1000),
      selection=BestSolutionSelection()
  )

  gen_alg.run()

  np.savetxt('genetical_clustering_accs.log', gen_clu.accs, fmt='%s')
  print('result', gen_alg.get_result())

  # gen_clu = GenCluster(DATA, 16)
  # np.savetxt('genetical_clustering_.log', gen_clu.pop_bin, fmt='%s')

  # print(get_comb(gen_clu.pop_count-1, (gen_clu.pop_count-1)/2))
  # exit()

  # for i in range(gen_clu.pop_size):

  #   print(i, gen_clu.pop[i])
  #   print(i, gen_clu.pop_bin[i])
  #   # print(i, get_set(rand_seed, gen_clu.pop[i][0][0], 15784))

  # # a = OneMax()
  # # print(a.create_solution())

  print('end')


class GenClust(BinaryProblem):
  def __init__(self, data, clust_min: int = None):
    super(GenClust, self).__init__()

    self.data = np.array(data)
    self.p = int(data.shape[0])

    self.clust_min = int(self.p * 0.20) if clust_min == None else clust_min
    self.clust_max = int(self.p - self.clust_min)

    self.k_min = 2
    self.k_max = int(self.p/self.clust_min)

    self.li_max_global = self.calc_li_max(0, self.k_max, [])
    self.idi_max_global = self.calc_idi_max(self.li_max_global, [])

    self.number_of_bits = self.calc_number_of_bits()
    self.number_of_objectives = 1
    self.number_of_variables = 1
    self.number_of_constraints = 0

    self.obj_directions = [self.MAXIMIZE]
    self.obj_labels = ['acc']

    self.accs = []

    self.print_info()

  def evaluate(self, solution: BinarySolution) -> BinarySolution:
    chrom = self.decode_chrom(solution.variables[0])
    k = len(chrom)

    #
    n = self.p
    indexes = list(range(1, self.p + 1))
    cluster_indexes = []

    for i in range(k):
      if (i + 1) == k:
        for index in indexes:
          cluster_indexes.append(
              [index, i + 1]
          )
        break

      [li, idi] = chrom[i]
      set = get_set(idi, n, li)

      for item in set:
        cluster_indexes.append(
            [indexes[item - 1], i + 1]
        )

      indexes = ft.reduce(
          lambda a, b: a if b in set else a + [b],
          indexes,
          []
      )

      n -= li

    #
    data_class = [0] * self.p
    for i in range(self.p):
      item = cluster_indexes[i]
      data_class[item[0]-1] = item[1]

    data = self.data[:, 1:]
    data = np.insert(data, 0, data_class, axis=1)

    #
    train_x, test_x, train_y, test_y = train_test_split(
        data[:, 1:], data[:, 0], test_size=0.5, random_state=rand_seed
    )

    tree_classifier = tree.DecisionTreeClassifier()
    tree_classifier.fit(train_x, train_y)
    pred_y = tree_classifier.predict(test_x)

    acc = metrics.accuracy_score(test_y, pred_y)

    for i in range(k):
      print('class ', i + 1, ':', (data[:, 0] == i+1).sum())
    print('ACC:', acc, 'k:', k)
    self.accs.append(acc)
    # print('RMSE:', metrics.mean_squared_error(test_y, pred_y))
    # print(metrics.confusion_matrix(test_y, pred_y))
    print('_________________\n')

    solution.objectives[0] = acc
    return solution

  def create_solution(self) -> BinarySolution:
    new_solution = BinarySolution(
        number_of_variables=1, number_of_objectives=1
    )
    new_solution.variables[0] = [
        (not rd.randint(0, 1)) for _ in range(self.number_of_bits)
    ]
    return new_solution

  def decode_chrom(self, bin_chrom):
    bin_chrom = [
        str(1 if bin_chrom[i] else 0) for i in range(self.number_of_bits)
    ]

    k = int(''.join(bin_chrom[:self.k_max.bit_length()]), 2)
    k = self.calc_wrap_k(k)

    bin_chrom = bin_chrom[self.k_max.bit_length():]

    gen_len = self.number_of_bits // self.k_max

    bin_chrom = [
        bin_chrom[j:j+gen_len]
        for j in range(0, self.number_of_bits, gen_len)
    ]

    chrom = []
    for i in range(k):
      gen = [
          bin_chrom[i][:self.li_max_global.bit_length()],
          bin_chrom[i][self.li_max_global.bit_length():]
      ]

      gen[0] = int(''.join(gen[0]), 2)
      li_max = self.calc_li_max(i, k, chrom)
      gen[0] = self.calc_wrap_li(gen[0], li_max)

      gen[1] = int(''.join(gen[1]), 2)
      idi_max = self.calc_idi_max(gen[0], chrom)
      gen[1] = gen[1] % idi_max

      chrom.append(gen)

    return chrom

  # calc
  def calc_number_of_bits(self):
    return (
        self.k_max.bit_length()
        +
        (self.li_max_global.bit_length() * self.k_max)
        +
        (self.idi_max_global.bit_length() * self.k_max)
    )

  def calc_li_max(self, i, k, chrom):
    return (
        self.p - ft.reduce(lambda a, b: a+b[0], chrom, 0)
    ) - (
        ((k-1) - i) * self.clust_min
    )

  def calc_idi_max(self, li_max, chrom):
    return get_comb(
        self.p - ft.reduce(lambda a, b: a+b[0], chrom, 0),
        li_max
    )

  def calc_wrap_k(self, k):
    return (k-self.k_min) % (self.k_max-self.k_min+1) + self.k_min

  def calc_wrap_li(self, li, li_max):
    return (li-self.clust_min) % (li_max-self.clust_min+1) + self.clust_min
  #

  def print_info(self):
    print(
        '***',
        f'p: {self.p}, k ∈ [2 …  {self.k_max}]',
        f'clust min: {self.clust_min}',
        f'clust max: {self.clust_max}',
        f'number of bits: {self.number_of_bits}',
        '***',
        sep='\n',
    )

  def get_name(self) -> str:
    return "GenClust"


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
    self.pop_bin = self.calc_pop_bin()

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
  def calc_li_max(self, i, k, chrom):
    return (
        self.pop_count - ft.reduce(lambda a, b: a+b[0], chrom, 0)
    ) - (
        ((k-1) - i) * self.pop_min
    )

  def calc_idi_max(self, li_max):
    return 5  # get_comb(self.pop_count, li_max)

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


def get_set(i, n, k):
  def C(n, k):
    result = 1
    for i in range(n):
      result *= (i+1)
    for i in range(k):
      result //= (i+1)
    for i in range(n-k):
      result //= (i+1)
    return result

  c = []
  r = i+0
  j = 0
  for s in range(1, k+1):
    cs = j+1
    while True:
      _c = C(n-cs, k-s)
      if not ((r - _c) > 0):
        break

      r -= _c
      cs += 1
    c.append(cs)
    j = cs
  return c


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