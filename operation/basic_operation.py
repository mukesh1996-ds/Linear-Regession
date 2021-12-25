"""
In this basic_operation we will preform all the basci operation such as checking the null value, shape of data,
apply describe functions.
"""

def check_null(data):
    """
    In this function you will get to know that is there any null value present in the data or not.
    This function is written by Mukesh Kumar
    Contact Email id is mks.mukesh1996@gmail.com
    """
    return data.isnull().sum()

def data_shape(data):
    """
        This function is used to check the shape of the data.
        This function is written by Mukesh Kumar
        Contact Email id is mks.mukesh1996@gmail.com
    """
    return data.shape

def check_describe(data):
    """
    [
        This function is used to check the following
        1. Mean, Median, Mode
        2. data count 
        3. precentile range
    ]
        This function is written by Mukesh
        contact email is mks.mukesh1996@gmail.com
    """
    return data.describe()

