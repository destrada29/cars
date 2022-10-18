
#import matplot to create graphs
import matplotlib.pyplot as plt
# its use to manage arrays
import numpy as np
#import pandas to read the dataset
import pandas as pd
# this is to create clusters or groups
from sklearn.cluster import KMeans

#reading the dataset and printing it 
cars=pd.read_csv("cars.csv")
cars=pd.DataFrame(data=cars)
print (cars)


#this is the info of the dataset variables
cars.info()

# I am counting if some of these variables are null
cars.isnull().sum()

# we have to transform the string variables into int variables, its because the model does not work with strings
dummies=pd.get_dummies(cars[' brand'])
cars_nuevo= cars.drop([' brand'],axis=1)
cars_nuevo = pd.concat([cars_nuevo, dummies], axis=1)
cars_nuevo.info()

#this import is to normalize the dataset or to scale all the values in a range of  0 to 1
from sklearn.preprocessing import MinMaxScaler

mm= MinMaxScaler()
cars_normalizado=mm.fit(cars_nuevo)
cars_normalizado= mm.transform(cars_nuevo)
cars_normalizado=pd.DataFrame(cars_normalizado)
cars_normalizado.describe()
cars_normalizado.head()





# I have to choose the number of groups to create, so I'm using the elbow method to select the number.
wcss=[]
for i in range(1,11):      
  kmeans = KMeans(n_clusters=i,max_iter=500)
  kmeans.fit(cars_normalizado)
  wcss.append(kmeans.inertia_)


plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("wcss")
plt.show()

#I am crating the clusters and training the model 
clustering=KMeans(n_clusters=4,max_iter=300)
clustering.fit(cars_normalizado)
clusters=pd.DataFrame(clustering.cluster_centers_)
print(clusters)


# I desnormalized the dataset to find the characteristics of the clusters 
cluster_desnor=mm.inverse_transform(clusters)
cluster_desnor=pd.DataFrame(cluster_desnor)
print(cluster_desnor)

# I am printing again the dataset but with a new colum calls KMeans_Clusters its contain the cluster that the row belogns to
cars['KMeans_Clusters']=clustering.labels_
cars.head()


# it is used to print the components 
from sklearn.decomposition import PCA

pca=PCA(n_components=3)
pca_credit=pca.fit_transform(cars_normalizado)
pca_credit_df=pd.DataFrame(data=pca_credit,columns=['component1','component2','component3'])
pca_add=pd.concat([pca_credit_df,cars['KMeans_Clusters']],axis=1)
pca_add.head()

grafica=plt.figure(figsize=(10,10))
graf=grafica.add_subplot(projection="3d")
graf.set_xlabel("component 1")
graf.set_ylabel("component 2")
graf.set_zlabel("component 3")
graf.set_title("Principal components")

colores=np.array(["blue","green","yellow","red"])

graf.scatter(xs=pca_add.Componente1,ys=pca_add.Componente2, zs=pca_add.Componente3, c=colores[pca_add.KMeans_Clusters],s=50)
plt.show()
