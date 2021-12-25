from sklearn.linear_model import LinearRegression

def linearRegression(x,y):
    lr_model = LinearRegression()
    lr_model.fit(x,y)
    predict_lr_model = lr_model.predict(x)
    return predict_lr_model