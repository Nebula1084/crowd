import pandas as pd

from nlp.sanitize import standardize


def load_excel(path, sheet, text_column, label_column):
    data = pd.ExcelFile(path)
    data = data.parse(sheet)
    data = standardize(data, text_column)
    return data['tokens'], data[label_column]


def load_csv(path, text_column, label_column):
    data = pd.read_csv(path, encoding='latin1')
    data = standardize(data, text_column)
    return data['tokens'], data[label_column]
