import numpy as np
import pandas as pd

def train_test_split(df: pd.DataFrame, test_size_ratio, seed) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    index = list(range(len(df)))
    np.random.shuffle(index)
    number_of_test_records = int(test_size_ratio*len(df))
    test_indexes = index[:number_of_test_records]
    df_test = df.iloc[test_indexes]
    df_train = df.drop(index=df.index[test_indexes])

    return df_train, df_test


def logistic_func(wage, features):
    big_number = 18
    # print(wage.shape)
    # print(features.shape)
    arg = np.dot(features, wage)
    arg = np.where(arg > big_number, big_number, np.where(arg < big_number, -1*big_number, arg))

    sigmoid = 1.0/(1 + np.exp(-arg))

    return sigmoid



def log_likelihood(wage, features, y_train, model):
    norm_wage = np.linalg.norm(wage)
    eps = 1e-10 #if prediction is 0 
    prediction = model(wage, features)
    result = np.sum(y_train*np.log(prediction + eps) + (1 - y_train)*np.log(1 - prediction))

    return result


def log_likelihood_derivative(wage, features, y_train, model):
    prediction = model(wage, features)
    delta_y = y_train - prediction
    result = np.dot(features.T, delta_y)

    return result


def classify(wage, features, model):
    result = model(wage, features)

    classification = np.where(result >= 0.5, 1, 0)

    return classification

