# Loading the required packages
from load_data.data import load_data
from operation.basic_operation import check_null , data_shape , check_describe
from outerlier.outerlier import outerlier_detection 
from vif.vif import standard_scaler,calculate_vif, convert_scaled_data_to_dataframe
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LinearRegression

"""
[ This is the stage 1 where the data is loaded into the model]
    
"""
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = load_data('G:\Machine Learning Project\model\linear_regression\data\housing.csv')
print(df)


"""
[ This is the stage 2 where general basic operation are done.]

"""
print(check_null(df))
print(data_shape(df))
print(check_describe(df))

"""
[This we are going to detect the percentage of outlier]
"""
print(outerlier_detection(df))

x = df.drop(columns = ['MEDV'])
print(x)
y = df['MEDV']
print(y)


print(standard_scaler(x))

data_ready = convert_scaled_data_to_dataframe(x)
print(calculate_vif(data_ready))

data_ready = data_ready.drop(columns = ['RAD', 'TAX'])
print(data_ready)

x_train,x_test,y_train,y_test = train_test_split(data_ready,y, test_size=0.2, random_state=55)

lr_model = LinearRegression()
lr_model.fit(x_train,y_train)
predict_lr_model = lr_model.predict(x_test)
print(predict_lr_model)
result = lr_model.score(x_test, y_test)
print(result)



# model dumping
filename = 'linear_model.pickle'
pickle.dump(lr_model, open(filename, 'wb'))


