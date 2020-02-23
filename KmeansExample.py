import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import mall dataset
dataset=pd.read_csv('Mall_Customers.csv')

#get annual income and spending score
X=dataset.iloc[:,[3,4]].values

#Elbow Methods

from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
	#fit values into KM
	kmeans=KMeans(n_clusters=i)
	kmeans.fit(X)
	wcss.append(kmeans.inertia_)


plt.plot(range(1,11),wcss)
plt.title("ELBOW METHOD")
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.plot()
plt.show()

#finding and visualizing
kmeans=KMeans(n_clusters=5)
y_kmeans=kmeans.fit_predict(X)

#print cluster labels
print ('cluster Labels',y_kmeans)

#visualizing the clusters
plt.scatter(X[:,0],X[:,1],c=y_kmeans,cmap='rainbow')

#Cluster center visualization
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='black',label='Centroid')

plt.title('Cluster of customers')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()

