#we are going to import the libraries
import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 

#we use pandas and scikit-learn library to load the dataset. The dataset can be loaded frm scikit-learn using the function load_boston() from the 
#scikit-learn datasets module

import pandas as pd
from sklearn.datasets import load_boston

dataset = load_boston()

#the four keywords used to access the dataset infomation is data,target, feature_names,DESCR.
print("[INFO] keys : {}".format(dataset.keys()))

print("[INFO] features shape : {}".format(dataset.data.shape))
print("[INFO] target shape   : {}".format(dataset.target.shape))

print("[INFO] dataset summary")
print(dataset.DESCR)

#correlation matrix that measures the linear relationships between the variables,heatmap function is used for this

correlation_matrix = boston.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)

#to check for the variations in the dataset, we plot a boxplot

boxplot=boston.boxplot(column=['CRIM', 'ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'
                                 ,'MEDV'])
boxplot.show()

