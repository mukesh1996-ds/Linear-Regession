import pandas as pd
import numpy as np

def load_data(x):
    """
    This function is used to load the data.
                        written by: Mukesh Kumar
                        Email id:  mks.mukesh1996@gmail.com
    """
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = pd.read_csv('G:\Machine Learning Project\model\linear_regression\data\housing.csv', header=None, delimiter=r"\s+", names=column_names)
    return df