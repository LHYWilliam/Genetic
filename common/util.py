import pandas as pd


def xlsx_to_csv(paths):
    for path in paths:
        data = pd.read_excel(path, sheet_name='Sheet1')
        with open(path.with_suffix('.csv'), 'w') as f:
            data.to_csv(f)


def csv_to_numpy(datas):
    datas = [data.to_numpy() for data in datas]

    return datas


def read_csvs(paths):
    datas = [pd.read_csv(path, encoding='gbk') for path in paths]

    return datas
