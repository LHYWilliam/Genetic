import os
import math
import random

import json
from pathlib import Path

import numpy as np
import pandas as pd

classes, days = 6, 7

# 成本上限
cost_stand = 1000
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

    #      品类名, 损耗率,         定价,             系数
    return names, AttritionRate, WholesalePrices, coefficients


def init_population(individual, population):
    # 初始化种群
    population = [np.zeros(individual) for _ in range(population)]
    for individual in population:
        # 随机初始化 总进货成本
        individual[0] = np.random.rand() * cost_stand
        # 随机初始化 进货成本相对比例    绝对比例 = 相对比例 / sum(相对比列s)
        individual[1:7] = [np.random.rand() for _ in range(classes)]
        # 随机初始化 定价
        individual[7:] = [np.random.rand() * pricing_stand for _ in range(days * classes)]

    # 编码
    population = [encode(individual) for individual in population]

    return population


def fitness_function(coefficients, individual, AttritionRate, WholesalePrices):
    # 解码
    individual = decode(individual)
    # 成本 定价
    cost, pricings = (individual[0] * np.array(individual[1:classes + 1] / np.sum(individual[1:classes + 1])),
                      np.array(individual[classes + 1:]).reshape(days, classes))

    sale_counts = np.zeros((days, classes))  # 销售量
    rest_counts = np.zeros((days, classes))  # 剩余量
    loss_counts = np.zeros((days, classes))  # 损耗量
    stock_counts = cost / WholesalePrices  # 进货量
    rest_counts[0] = stock_counts

    for day in range(days):
        # 销售量 由方程计算
        sale_counts[day] = np.array(
            [np.sum([one * math.pow(pricing, i) for i, one in enumerate(coefficient)]) for coefficient, pricing in
             zip(coefficients, pricings[day])])
        # 销售量 = min(销售量, 剩余量)
        sale_counts[day] = np.array(
            [max(min(sale, rest), 0) for sale, rest in zip(sale_counts[day], rest_counts[day])])
        # 剩余量 = 剩余量 - 卖出量
        rest_counts[day] -= sale_counts[day]
        # 损耗量 = 损耗率 * 损耗量
        loss_counts[day] = AttritionRate * rest_counts[day]
        # 剩余量 = 剩余量 - 损耗量
        rest_counts[day] -= loss_counts[day]
        # 后一天的 初始剩余量 为 前一天的 结束剩余量
        if day != (days - 1):
            rest_counts[day + 1] = rest_counts[day]

    # 销售额 = 销售量 * 定价
    sales = sale_counts * pricings
    # 利润 = 销售量 - 成本
    profit = np.sum(sales) - np.sum(cost)

    # 过程记录
    details = {'profit': profit,
               'sale_counts': sale_counts,
               'rest_counts': rest_counts,
               'loss_counts': loss_counts}

    return details


def crossover(population, fitness, crossover_rate):
    # 适应概率
    probabilities = [one / np.sum(fitness) for one in fitness]
    # 根据适应概率选取父本
    parents = random.choices(population, probabilities, k=int(len(population) * crossover_rate * 2))
    parents = [parent.split(' ') for parent in parents]

    # 确定交叉点
    point = np.random.randint(1, len(parents[0]) - 1)
    # 交叉互换产生子代
    children = [' '.join(parent1[:point] + parent2[point:]) for parent1, parent2 in zip(parents[0::2], parents[1::2])]

    return children


def mutate(children, mutation_rate):
    for i in range(len(children)):
        # 若随机概率小于变异概率，则变异
        if np.random.rand() < mutation_rate:
            children[i] = children[i].split(' ')
            choice = np.random.rand()
            if choice < 0.2:  # 变异 总成本
                children[i][0] = encode([np.random.rand() * cost_stand])
            elif choice < 0.4:  # 变异 成本相对比例
                children[i][np.random.randint(1, classes + 1)] = encode([np.random.rand()])
            else:  # 变异 定价
                children[i][np.random.randint(len(children[i]) - classes - 1) + classes + 1] \
                    = encode([np.random.rand() * pricing_stand])
            children[i] = ' '.join(children[i])

    return children


def select(population, fitness, top):
    # 捆绑适应度与群体
    fitness_population = [{'fitness': fitness, 'individual': individual}
                          for fitness, individual in zip(fitness, population)]

    # 根据适应度对群体排序，选择前top名
    selected = [one['individual'] for one in
                sorted(fitness_population, key=lambda one: one['fitness'], reverse=True)[:top]]

    return selected


def encode(individual):
    # 编码
    binary = ' '.join([format(np.float64(one).view(np.int64), f'0{64}b') for one in individual])

    return binary


def decode(binary):
    # 解码
    individual = np.array([np.int64(int(one, 2)).view(np.float64) for one in binary.split(' ')])

    return individual


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
