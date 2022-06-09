import math
import random as rd
import numpy as np
import functools as ft

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from jmetal.operator import SPXCrossover, BitFlipMutation, BinaryTournamentSelection
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
      offspring_population_size=80,
      mutation=BitFlipMutation(0.02),
      crossover=SPXCrossover(0.8),
      termination_criterion=StoppingByEvaluations(10000),
      selection=BinaryTournamentSelection()
  )

  gen_alg.run()

  np.savetxt('genetical_clustering_accs.log', gen_clu.accs, fmt='%s')
  print('result', gen_alg.get_result())

  print('end')


class GenClust(BinaryProblem):
  def __init__(self, data, clust_min: int = None):
    super(GenClust, self).__init__()
    self.data = data
    self.data_pos = [row for row in data if row[0] == 1]
    self.data_neg = [row for row in data if row[0] == 0]

    def calc_data_info(data, bit_start):
      p = int(len(data))

      clust_min = int(p * 0.25)
      clust_max = int(p - clust_min)

      k_min = 2
      k_max = int(p/clust_min)

      li_max_global = self.calc_li_max(p, clust_min, 0, k_max, [])
      idi_max_global = self.calc_idi_max(p, li_max_global, [])

      info = {
          'p': p,
          'clust_min': clust_min,
          'clust_max': clust_max,
          'k_min': k_min,
          'k_max': k_max,
          'li_max_global': li_max_global,
          'idi_max_global': idi_max_global,
          'bit_start': bit_start,
      }

      info['bit_end'] = self.calc_number_of_bits_indiv(info)
      info['bit_end'] += info['bit_start']

      return info

    self.pos = calc_data_info(self.data_pos, 0)
    self.neg = calc_data_info(self.data_neg, self.pos['bit_end'])

    self.number_of_bits = self.calc_number_of_bits()
    self.number_of_objectives = 1
    self.number_of_variables = 1
    self.number_of_constraints = 0

    self.obj_directions = [self.MINIMIZE]
    self.obj_labels = ['acc']

    self.accs = []
    self.best_acc = 0
    self.eva_n = 0

    self.print_info()

  def evaluate(self, solution: BinarySolution) -> BinarySolution:
    (chrom_pos, chrom_neg) = self.decode_chrom(solution.variables[0])

    #
    data_class_pos = self.gen_data_class('pos', chrom_pos)
    data_class_neg = self.gen_data_class('neg', chrom_neg)

    data_class = data_class_neg + data_class_pos

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

    self.eva_n += 1
    if acc > self.best_acc:
      self.best_acc = acc

    print(f'evaluation N:{self.eva_n}')

    for i in range(len(chrom_pos) + len(chrom_neg)):
      print('- class ', i + 1, ':', (data[:, 0] == i+1).sum())

    print('ACC:', acc)
    print('B ACC:', self.best_acc)
    self.accs.append(acc)
    print('_________________\n')

    solution.objectives[0] = acc * -1
    return solution

  def gen_data_class(self, type, chrom):
    info = self.pos if type == 'pos' else self.neg

    k = len(chrom)
    n = info['p']
    indexes = list(range(1, info['p'] + 1))
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

    data_class = [0 if type == 'pos' else 2] * info['p']
    for i in range(info['p']):
      item = cluster_indexes[i]
      data_class[item[0]-1] += item[1]

    return data_class

  def decode_chrom(self, bin_chrom):
    bin_chrom = [
        str(1 if bin_chrom[i] else 0) for i in range(self.number_of_bits)
    ]

    chrom_pos = self.decode_chrom_indiv('pos', bin_chrom)
    chrom_neg = self.decode_chrom_indiv('neg', bin_chrom)

    return (chrom_pos, chrom_neg)

  def decode_chrom_indiv(self, type, bin_chrom):
    info = self.pos if type == 'pos' else self.neg

    bin_chrom = bin_chrom[info['bit_start']:info['bit_end']]

    k = int(''.join(bin_chrom[:info['k_max'].bit_length()]), 2)
    k = self.calc_wrap_k(info['k_min'], info['k_max'], k)

    bin_chrom = bin_chrom[info['k_max']:]

    bin_len = len(bin_chrom)
    gen_len = bin_len // info['k_max']

    bin_chrom = [
        bin_chrom[j:j+gen_len]
        for j in range(0, bin_len, gen_len)
    ]

    chrom = []
    for i in range(k):
      gen = [
          bin_chrom[i][:info['li_max_global'].bit_length()],
          bin_chrom[i][info['li_max_global'].bit_length():]
      ]

      gen[0] = int(''.join(gen[0]), 2)
      li_max = self.calc_li_max(info['p'], info['clust_min'], i, k, chrom)
      gen[0] = self.calc_wrap_li(info['clust_min'], gen[0], li_max)

      gen[1] = int(''.join(gen[1]), 2)
      idi_max = self.calc_idi_max(info['p'], gen[0], chrom)
      gen[1] = gen[1] % idi_max

      chrom.append(gen)

    return chrom

  def create_solution(self) -> BinarySolution:
    new_solution = BinarySolution(
        number_of_variables=1, number_of_objectives=1
    )
    new_solution.variables[0] = [
        (not rd.randint(0, 1)) for _ in range(self.number_of_bits)
    ]
    return new_solution

  # calc
  def calc_number_of_bits(self):
    pos_len = self.calc_number_of_bits_indiv(self.pos)
    neg_len = self.calc_number_of_bits_indiv(self.neg)
    return pos_len + neg_len

  def calc_number_of_bits_indiv(self, info):
    return (
        info['k_max'].bit_length()
        +
        (info['li_max_global'].bit_length() * info['k_max'])
        +
        (info['idi_max_global'].bit_length() * info['k_max'])
    )

  def calc_li_max(self, p, clust_min, i, k, chrom):
    return (
        p - ft.reduce(lambda a, b: a+b[0], chrom, 0)
    ) - (
        ((k-1) - i) * clust_min
    )

  def calc_idi_max(self, p, li_max, chrom):
    return get_comb(
        p - ft.reduce(lambda a, b: a+b[0], chrom, 0),
        li_max
    )

  def calc_wrap_k(self, k_min, k_max,  k):
    return (k-k_min) % (k_max-k_min+1) + k_min

  def calc_wrap_li(self, clust_min,  li, li_max):
    return (li-clust_min) % (li_max-clust_min+1) + clust_min
  #

  def print_info(self):
    print(
        '***',
        # f'p: {self.p}, k ∈ [2 …  {self.k_max}]',
        # f'clust min: {self.clust_min}',
        # f'clust max: {self.clust_max}',
        f'number of bits: {self.number_of_bits}',
        '***',
        sep='\n',
    )

  def get_name(self) -> str:
    return "GenClust"

#


def get_set(i, n, k):
  c = []
  r = i+0
  j = 0
  for s in range(1, k+1):
    cs = j+1
    while True:
      _c = get_comb(n-cs, k-s)
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


#
#
if __name__ == '__main__':
  main()
