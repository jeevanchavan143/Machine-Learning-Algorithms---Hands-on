import pandas as pd
import numpy as np
dataset=pd.read_csv('winequality-white.csv')
print dataset.head()
print "shape", dataset.shape
x=dataset.drop("quality",axis=1)
y=dataset["quality"]
#choosing input and output
x.shape
y.shape

#spliting of dataset
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifierDT=DecisionTreeClassifier()
classifierDT.fit(x_train,y_train)
y_predDTC=classifierDT.predict(x_test)


#Classification Performance
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print accuracy_score(y_test,y_predDTC)
print confusion_matrix(y_test,y_predDTC)
print classification_report(y_test,y_predDTC)
#sample data
#new=np.array([[6.2,0.23,0.35,0.7,0.051,24,111,0.9916,3.37,0.43,11]])
new=np.array([[6.2,0.23,0.35,0.7,0.051,24,111,0.9916,3.37,0.43,11]])
y_predTest1=classifierDT.predict(new)
print "predicted class is",y_predTest1[0]


#Decision Tree Image Generation
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
dot_data=StringIO()
export_graphviz(classifierDT,out_file=dot_data,filled=True,rounded=True,special_characters=True)
import pydotplus
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("WineQualityDT.png")
#image Saved

#######
####
#decision tree Regressor

from sklearn.tree import DecisionTreeRegressor
regressorDT=DecisionTreeRegressor()
regressorDT.fit(x_train,y_train)
y_predDTR=regressorDT.predict(x_test)
#dataframe
df=pd.DataFrame({'Actual':y_test,'Predicted':y_predDTR})
print df

#characterizing the Regressor
from sklearn import metrics
print 'Mean Absolute Error'
print metrics.mean_absolute_error(y_test,y_predDTR)
print 'mean_squared_error'
metrics.mean_squared_error(y_test,y_predDTR)
print 'root mean squared Error '
np.sqrt(metrics.mean_squared_error(y_test,y_predDTR))
print 'mean of quality:'
dataset.quality.mean()
#compre mean and mean absolute error if absolute mean  less than 10% of  then best algorithm works fine


#Random Forest Classifier
 
from sklearn.ensemble import RandomForestClassifier
classifierRF=RandomForestClassifier(n_estimators=10)
classifierRF.fit(x_train,y_train)
y_predRFC=classifierRF.predict(x_test)
df=pd.DataFrame({'Actual':y_test,'Predicted':y_predRFC})
print df

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print confusion_matrix(y_test,y_predRFC)
print accuracy_score(y_test,y_predRFC)
print classification_report(y_test,y_predRFC)

#For 100 Trees
classifierRF=RandomForestClassifier(n_estimators=100)
classifierRF.fit(x_train,y_train)
y_predRFC=classifierRF.predict(x_test)
df=pd.DataFrame({'Actual':y_test,'Predicted':y_predRFC})
print accuracy_score(y_test,y_predRFC)
print confusion_matrix(y_test,y_predRFC)

#for 1000 trees

classifierRF=RandomForestClassifier(n_estimators=1000)
classifierRF.fit(x_train,y_train)
y_predRFC=classifierRF.predict(x_test)
print accuracy_score(y_test,y_predRFC)

#for 2000 trees
classifierRF=RandomForestClassifier(n_estimators=2000)
classifierRF.fit(x_train,y_train)
y_predRFC=classifierRF.predict(x_test)

print accuracy_score(y_test,y_predRFC)
print confusion_matrix(y_test,y_predRFC)
print classification_report(y_test,y_predRFC)


 
#Random Forest Regressor
#for 10 trees 
from sklearn.ensemble import RandomForestRegressor
regressorRF=RandomForestRegressor(n_estimators=10)
regressorRF.fit(x_train,y_train)
y_predRFR=regressorRF.predict(x_test)

#for 100 trees and so on
regressorRF=RandomForestRegressor(n_estimators=100)
regressorRF.fit(x_train,y_train)
y_predRFR=regressorRF.predict(x_test)

#for 1000 trees
regressorRF=RandomForestRegressor(n_estimators=1000)
regressorRF.fit(x_train,y_train)
y_predRFR=regressorRF.predict(x_test)

#for 2000 trees 
regressorRF=RandomForestRegressor(n_estimators=2000)
regressorRF.fit(x_train,y_train)
y_predRFR=regressorRF.predict(x_test)
#characterizing the Regressor
from sklearn import metrics
print 'Mean Absolute Error'
print metrics.mean_absolute_error(y_test,y_predRFR)
print 'mean_squared_error'
metrics.mean_squared_error(y_test,y_predRFR)
print 'root mean squared Error '
np.sqrt(metrics.mean_squared_error(y_test,y_predRFR))
print 'mean of quality:'
dataset.quality.mean()
#compre mean and mean absolute error if absolute mean  less than 10% of  then best algorithm works fine


