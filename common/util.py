import os
import math
import random

import yaml
from pathlib import Path

import numpy as np
import pandas as pd

# 进货量上限
stock_stand = 10000
# 进价上限
pricing_stand = 50


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
    with open(Path('data/coefficients.yaml'), 'r') as f:
        coefficients = yaml.safe_load(f)

    # 品类名, 损耗率, 定价, 系数
    return names, AttritionRate, WholesalePrices, coefficients


def init_population(individual, population):
    # 初始化种群
    population = [np.zeros(individual) for _ in range(population)]
    for individual in population:
        # 随机初始化进货量
        individual[:len(individual) // 2] = [np.random.rand() * stock_stand for _ in range(len(individual) // 2)]
        # 随机初始化定价
        individual[len(individual) // 2:] = [np.random.rand() * pricing_stand for _ in range(len(individual) // 2)]

    return population


def fitness_function(coefficients, individual, AttritionRate, WholesalePrices):
    # 进货量, 定价
    stock_counts, pricings = individual[:len(individual) // 2], individual[len(individual) // 2:]

    # 卖出量
    sale_counts = np.array([sum([one * math.pow(pricing, i) for i, one in enumerate(coefficient)])
                            for coefficient, pricing in zip(coefficients, pricings)])
    # 损耗量 = 损耗率 * 进货量
    loss_counts = AttritionRate * stock_counts
    # 卖出量 = min(卖出量, 进货量 - 损耗量)
    sale_counts = np.array([min(sale, stock) for sale, stock in zip(sale_counts, stock_counts - loss_counts)])

    # 成本 = 进货量 * 进价
    cost = stock_counts * WholesalePrices
    # 销售量 = 卖出量 * 定价
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
    cross_point = np.random.randint(1, len(parents[0]))
    # 交叉互换产生子代
    children = [np.hstack((parent1[:cross_point], parent2[cross_point:]))
                for parent1, parent2 in zip(parents[0::2], parents[1::2])]

    return children


def mutate(children, mutation_rate):
    for child in children:
        # 若随机概率小于变异概率，则变异
        if np.random.rand() < mutation_rate:
            # 随机变异点
            point = np.random.randint(len(child))
            if point < len(child) // 2:
                child[point] = np.random.rand() * stock_stand
            else:
                child[point] = np.random.rand() * pricing_stand

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
