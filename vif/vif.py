from statsmodels.stats.outliers_influence import variance_inflation_factor
#from load_data.data import load_data
from sklearn.preprocessing import StandardScaler


import pandas as pd
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('G:\Machine Learning Project\model\linear_regression\data\housing.csv', names = column_names, header=None, delimiter=r"\s+" )
print((df))

"""
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = load_data('G:\Machine Learning Project\model\linear_regression\data\housing.csv', names = column_names,header=None, delimiter=r"\s+")
print(df)
"""
x = df.drop(columns = ['MEDV'])
print(x)
y = df['MEDV']
print(y)

def standard_scaler(x):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled

""" [Testing]"""
scaled_x = standard_scaler(x)
print(scaled_x)
print(standard_scaler(x))


# converting array to data frame
def convert_scaled_data_to_dataframe(x):
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    x_dataframe = pd.DataFrame(scaled_x, columns = column_names)
    return x_dataframe 

""" [Testing]"""
data_ready = convert_scaled_data_to_dataframe(scaled_x)
print(data_ready)

def calculate_vif(x):
    vif = pd.DataFrame()
    vif['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif['variable'] = x.columns
    return vif

""" [Testing]"""
print(calculate_vif(data_ready))


