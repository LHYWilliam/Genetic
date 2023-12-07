import os
import math
import yaml
from pathlib import Path

import numpy as np
import pandas as pd


def load_data():
    xlsx_paths = [Path('data/' + path) for path in os.listdir('data') if Path(path).suffix == '.xlsx']
    xlsx_to_csv([path for path in xlsx_paths if path.suffix == '.xlsx'])

    csv_paths = [Path('data/' + path) for path in os.listdir('data') if Path(path).suffix == '.csv']
    datas = read_csv([path for path in csv_paths if path.suffix == '.csv'])

    AttritionRate, WholesalePrices = csv_to_numpy(datas)

    AttritionRate = {name: rate for _, name, rate in AttritionRate}
    WholesalePrices = {name: price for _, name, price in WholesalePrices}

    names = list(AttritionRate.keys())

    with open(Path('data/coefficients.yaml'), 'r') as f:
        coefficients = yaml.safe_load(f)

    return names, AttritionRate, WholesalePrices, coefficients


def calculate_sales(coefficients, pricing):
    sales = [sum([one * math.pow(pricing, i + 1) for i, one in enumerate(coefficient)])
             for coefficient, pricing in zip(coefficients, pricing)]

    return sales


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
