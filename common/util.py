import os
import math
import random

import json
from pathlib import Path

import numpy as np
import pandas as pd

# 成本上限
cost_stand = 4000
# 定价上限
pricing_stand = 20


def load_data():
    # 获取data中xlsx路径
    xlsx_paths = [Path('data/' + path) for path in os.listdir('data') if Path(path).suffix == '.xlsx']
    # 将xlsx转化为csv
    xlsx_to_csv(xlsx_paths)

    # 获取data中csv路径
    csv_paths = [Path('data/' + path) for path in os.listdir('data') if Path(path).suffix == '.csv']
    # 读取csv
    datas = read_csv(csv_paths)

    # 损耗率, 定价
    AttritionRate, WholesalePrices = csv_to_numpy(datas)

    # {品类名: 损耗率}
    AttritionRate = {name: rate for _, name, rate in AttritionRate}
    # {品类名: 定价}
    WholesalePrices = {name: price for _, name, price in WholesalePrices}

    # 品类名
    names = list(AttritionRate.keys())

    # 读取方程系数
    with open(Path('data/coefficients.json'), 'r') as f:
        coefficients = json.load(f)

    # 品类名, 损耗率, 定价, 系数
    return names, AttritionRate, WholesalePrices, coefficients


def init_population(individual, population):
    # 初始化种群
    population = [np.zeros(individual) for _ in range(population)]
    for individual in population:
        # 随机初始化总进货成本
        individual[0] = np.random.rand() * cost_stand
        # 随机初始化进货成本比例
        individual[1:7] = list(np.random.dirichlet(np.ones(6)))
        # 随机初始化定价
        individual[7:] = [np.random.rand() * pricing_stand for _ in range(len(individual) - 7)]

    return population


def fitness_function(coefficients, individual, AttritionRate, WholesalePrices):
    # 品类进货成本, 定价
    cost, pricings = individual[0] * np.array(individual[1:7]), np.array(individual[7:])

    stock_counts = cost / WholesalePrices  # 进货量
    rest_counts = stock_counts[:]  # 剩余量
    sale_counts = np.zeros(6 * 7)  # 销售量

    for day in range(7):
        # 销售量
        sale_counts[day * 6:(day + 1) * 6] = np.array(
            [sum([one * math.pow(pricing, i) for i, one in enumerate(coefficient)])
             for coefficient, pricing in zip(coefficients, pricings[day * 6:(day + 1) * 6])])
        # 销售量 = min(销售量, 剩余量)
        sale_counts[day * 6:(day + 1) * 6] = np.array([max(min(sale, rest), 0)
                                                       for sale, rest in
                                                       zip(sale_counts[day * 6:(day + 1) * 6], rest_counts)])
        # 剩余量 = 剩余量 - 卖出量
        rest_counts -= sale_counts[day * 6:(day + 1) * 6]

        loss_counts = AttritionRate * rest_counts
        # 剩余量 = 剩余量 - 损耗率 * 损耗量
        rest_counts -= AttritionRate * loss_counts

    # 销售额 = 销售量 * 定价
    sales = sale_counts * pricings
    # 利润 = 销售量 - 成本
    profit = sum(sales) - sum(cost)

    return profit


def crossover(population, fitness, crossover_rate):
    # 适应概率
    probabilities = [one / sum(fitness) for one in fitness]
    # 根据适应概率选取父本
    parents = random.choices(population, probabilities, k=int(len(population) * crossover_rate * 2))

    # 确定交叉点
    point = np.random.randint(1, len(parents[0]))
    # 交叉互换产生子代
    children = [np.hstack((parent1[:point], parent2[point:]))
                for parent1, parent2 in zip(parents[0::2], parents[1::2])]

    return children


def mutate(children, mutation_rate):
    for child in children:
        for _ in range(len(children[0])):
            # 若随机概率小于变异概率，则变异
            if np.random.rand() < mutation_rate:
                choice = np.random.randint(10)
                if choice in (0, 1):
                    child[0] = np.random.rand() * cost_stand
                elif choice in (2, 3):
                    child[1:7] = list(np.random.dirichlet(np.ones(6)))
                else:
                    child[np.random.randint(len(child) - 7) + 7] = np.random.rand() * pricing_stand

    return children


def select(population, fitness, top):
    # 捆绑适应度与群体
    fitness_individual = [(fitness, individual) for fitness, individual in zip(fitness, population)]
    # 根据适应度对群体排序，选择前top名
    selected = [one[1] for one in sorted(fitness_individual, key=lambda x: -x[0])[:top]]

    return selected


def xlsx_to_csv(paths):
    for path in paths:
        data = pd.read_excel(path)
        with open(path.with_suffix('.csv'), 'w') as f:
            data.to_csv(f)


def csv_to_numpy(datas):
    datas = [data.to_numpy() for data in datas]

    return datas


def read_csv(paths):
    datas = [pd.read_csv(path, encoding='gbk') for path in paths]

    return datas
