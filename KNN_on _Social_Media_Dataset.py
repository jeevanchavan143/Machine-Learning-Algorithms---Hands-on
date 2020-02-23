import pandas as pd
import numpy as np

dataset=pd.read_csv("Social_Network_Ads.csv")
dataset.head()
dataset.shape

X=dataset.iloc[:,2:4].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print confusion_matrix(y_test,y_pred)
print accuracy_score(y_test,y_pred)*100

new=np.array([[32,74000]])
print new
y_prednew=classifier.predict(new)
print y_prednew

#After using StandardScaler

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print confusion_matrix(y_test,y_pred)
print 'Accuracy Score',accuracy_score(y_test,y_pred)*100

