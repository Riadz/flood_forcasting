import math
import random as rd
import numpy as np
import functools as ft

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from jmetal.operator import SPXCrossover, BitFlipMutation, NullMutation, BestSolutionSelection, BinaryTournamentSelection, crossover
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.operator import Crossover, Mutation
from jmetal.util.ckecking import Check
from typing import List
import copy

rand_seed = 88
np.random.seed(rand_seed)
rd.seed(rand_seed)


def main():
  DATA = np.genfromtxt(
      r'./data/Data_vietnam.csv', delimiter=','
  )[1:]

  gen_clu = GenClust(DATA)

  gen_alg = NSGAII(
      problem=gen_clu,
      population_size=100,
      offspring_population_size=80,
      mutation=CustomMutation(0.02, gen_clu),
      crossover=CustomCrossover(0.8, gen_clu),
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

    self.obj_directions = [self.MINIMIZE]
    self.obj_labels = ['acc']

    self.accs = []
    self.best_acc = 0
    self.eva_n = 0

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
    sims = []
    for i in range(k):
      data_k = [row for row in data if row[0] == i+1]

      sim = cosine_similarity(data_k, data_k)
      sim = np.mean(sim)
      sims.append(sim)

    acc = np.average(sims)
    print(sims, acc)

    self.eva_n += 1
    if acc > self.best_acc:
      self.best_acc = acc

    #
    train_x, test_x, train_y, test_y = train_test_split(
        data[:, 1:], data[:, 0], test_size=0.5, random_state=rand_seed
    )

    tree_classifier = tree.DecisionTreeClassifier()
    tree_classifier.fit(train_x, train_y)
    pred_y = tree_classifier.predict(test_x)

    print('* acc: ', metrics.accuracy_score(test_y, pred_y))

    print(f'evaluation N:{self.eva_n}')
    for i in range(k):
      print('class ', i + 1, ':', (data[:, 0] == i+1).sum())
    print('ACC:', acc, 'k:', k)
    print('B ACC:', self.best_acc)
    self.accs.append(acc)
    # print('RMSE:', metrics.mean_squared_error(test_y, pred_y))
    # print(metrics.confusion_matrix(test_y, pred_y))
    print('_________________\n')

    solution.objectives[0] = acc * -1
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


class CustomCrossover(Crossover[BinarySolution, BinarySolution]):

  def __init__(self, probability: float, problem: GenClust):
    self.problem = problem
    super(CustomCrossover, self).__init__(probability=probability)

  def execute(self, parents: List[BinarySolution]) -> List[BinarySolution]:
    Check.that(type(parents[0]) is BinarySolution, "Solution type invalid")
    Check.that(type(parents[1]) is BinarySolution, "Solution type invalid")
    Check.that(len(parents) == 2,
               'The number of parents is not two: {}'.format(len(parents)))

    offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
    rand = rd.random()

    if rand <= self.probability:
      # 1. Get the total number of bits
      d_parents = [
          self.problem.decode_chrom(parents[0].variables[0]),
          self.problem.decode_chrom(parents[1].variables[0])
      ]
      k_max_cross = max(len(d_parents[0]), len(d_parents[1]))
      total_number_of_bits = (
          self.problem.k_max.bit_length()
          +
          (self.problem.li_max_global.bit_length() * k_max_cross)
          +
          (self.problem.idi_max_global.bit_length() * k_max_cross)
      )

      # 2. Calculate the point to make the crossover
      crossover_point = rd.randrange(0, total_number_of_bits)

      # 3. Compute the variable containing the crossover bit
      variable_to_cut = 0
      bits_count = len(parents[1].variables[variable_to_cut])
      while bits_count < (crossover_point + 1):
        variable_to_cut += 1
        bits_count += len(parents[1].variables[variable_to_cut])

      # 4. Compute the bit into the selected variable
      diff = bits_count - crossover_point
      crossover_point_in_variable = len(
          parents[1].variables[variable_to_cut]) - diff

      # 5. Apply the crossover to the variable
      bitset1 = copy.copy(parents[0].variables[variable_to_cut])
      bitset2 = copy.copy(parents[1].variables[variable_to_cut])

      for i in range(crossover_point_in_variable, len(bitset1)):
        swap = bitset1[i]
        bitset1[i] = bitset2[i]
        bitset2[i] = swap

      offspring[0].variables[variable_to_cut] = bitset1
      offspring[1].variables[variable_to_cut] = bitset2

      # 6. Apply the crossover to the other variables
      for i in range(variable_to_cut + 1, parents[0].number_of_variables):
        offspring[0].variables[i] = copy.deepcopy(
            parents[1].variables[i])
        offspring[1].variables[i] = copy.deepcopy(
            parents[0].variables[i])

    return offspring

  def get_number_of_parents(self) -> int:
    return 2

  def get_number_of_children(self) -> int:
    return 2

  def get_name(self) -> str:
    return 'Single point crossover'


class CustomMutation(Mutation[BinarySolution]):

  def __init__(self, probability: float, problem: GenClust):
    self.problem = problem
    super(CustomMutation, self).__init__(probability=probability)

  def execute(self, solution: BinarySolution) -> BinarySolution:
    Check.that(type(solution) is BinarySolution, "Solution type invalid")

    for i in range(solution.number_of_variables):
      chrom = self.problem.decode_chrom(solution.variables[i])
      k_max_mut = len(chrom)
      total_number_of_bits = (
          self.problem.k_max.bit_length()
          +
          (self.problem.li_max_global.bit_length() * k_max_mut)
          +
          (self.problem.idi_max_global.bit_length() * k_max_mut)
      )

      for j in range(total_number_of_bits):
        rand = rd.random()
        if rand <= self.probability:
          solution.variables[i][j] = True if solution.variables[i][j] is False else False

    return solution

  def get_name(self):
    return 'BitFlip mutation'

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
