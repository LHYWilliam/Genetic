import json
import random

import numpy as np
import pandas as pd

from common.util import (load_data, init_population, fitness_function, crossover, mutate, select)

classes, days = 6, 7

if __name__ == '__main__':
    """
    有 初始化种群 与 载入种群 两个模式供选择
    
    初始化训练时，建议将 util.py 中的 cost_stand 成本上限 从 1000 开始逐渐增加
    否则练速度很慢
    
    定价中可能存在无效数据
    原因是某一商品的的剩余量为 0 以后 该商品的定价不对该个体的 适应度 产生影响
    """
    random.seed(0)
    np.random.seed(0)

    # 品类名, 损耗率, 定价, 方程系数
    names, AttritionRate, WholesalePrices, coefficients = load_data()
    # 将字典转化为numpy
    AttritionRate = np.array(list(AttritionRate.values()))
    WholesalePrices = np.array(list(WholesalePrices.values()))
    coefficients = [np.array(coefficient) for coefficient in coefficients.values()]

    population_size = 10000  # 种群大小
    generations = 1024  # 迭代代数
    crossover_rate = (population_size - population_size // 10) / population_size  # 重组概率
    mutation_rate = 0.5  # 变异概率

    # 初始化种群
    log = {}  # 数据记录
    population = init_population(1 + classes + classes * days, population_size, WholesalePrices)

    # # 载入种群
    # with open('result/log.json', 'r') as f:
    #     log = json.load(f)
    # with open('result/best.json', 'r') as f:
    #     best = encode(json.load(f))
    # population = [best[:] for _ in range(population_size)]

    # 迭代开始
    already = len(log)
    for generation in range(generations):
        generation = generation + already
        # 计算适应度(利润)
        details = [fitness_function(coefficients, individual, AttritionRate, WholesalePrices)
                   for individual in population]

        # 处理数据
        fitness = [individual['profit'] for individual in details]
        best = sorted(details, key=lambda individual: individual['profit'], reverse=True)[0]

        print(f'\rgeneration {generation + 1}: {best["profit"]}', end='')

        individual, sale_count, rest_count, loss_count = \
            best['individual'], best['sale_counts'], best['rest_counts'], best['loss_counts']
        cost, proportion, pricing = (np.array(individual[0]), np.array(individual[1:1 + classes]),
                                     np.array(individual[1 + classes:]).reshape(days, classes))

        # 组织数据
        # 历史数据
        log[f'generation {generation + 1}'] = {'fitness': best['profit'], 'individual': list(individual)}
        profits = {'generation': [generation + 1 for generation in range(generation + 1)],
                   'fitness': [one['fitness'] for one in log.values()]}

        # 本次数据
        result = dict({'成本': cost, '比例': proportion},
                      **{f'第{day + 1}天定价': pricing for day, pricing in enumerate(pricing)})
        sale_count = {f'第{day + 1}天销售量': sale_count for day, sale_count in enumerate(sale_count)}
        rest_count = {f'第{day + 1}天剩余量': rest_count for day, rest_count in enumerate(rest_count)}
        loss_count = {f'第{day + 1}天损耗量': loss_count for day, loss_count in enumerate(loss_count)}

        # 保存数据
        with open('result/best.json', 'w') as f:
            json.dump(list(best['individual']), f, indent=2)
        with open('result/log.json', 'w') as f:
            json.dump(log, f, indent=2)

        pd.DataFrame(result).to_excel('result/result.xlsx')
        pd.DataFrame(sale_count).to_excel('result/sale_count.xlsx')
        pd.DataFrame(rest_count).to_excel('result/rest_count.xlsx')
        pd.DataFrame(loss_count).to_excel('result/loss_count.xlsx')
        pd.DataFrame(profits).to_excel('result/log.xlsx')

        # 重组
        children = crossover(population, fitness, crossover_rate)
        # 变异
        children = mutate(children, mutation_rate, WholesalePrices)
        # 选择
        parents = select(population, fitness, top=population_size - len(children))

        # 获得下一代种群
        population = children + parents
