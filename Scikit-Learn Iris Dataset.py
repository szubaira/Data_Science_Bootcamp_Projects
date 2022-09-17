#!/usr/bin/env python
# coding: utf-8

# 

# # Exercises

# **Complete every exercise in the Basic and Visualization, K-Nearest Neighbors Algorithm, and Logistic Regression sections.  Put every solution in its own file and push a folder containing all of your code to a Github repo. Submit the URL for that repo.**

# ### [Python Machine Learning Iris flower Data Set](https://www.w3resource.com/machine-learning/scikit-learn/iris/index.php)

# Scikit-learn is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
# 
# The best way we learn anything is by practice and exercise questions. We have started this section for those (beginner to intermediate) who are familiar with Python, Scikit-learn. Hope, these exercises help you to improve your Machine learning skills using Scikit-learn. Currently, data set are available, we are working hard to add more exercises.

# #### Iris Flower Data Set

# From Wikipedia - The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. Two of the three species were collected in the Gaspé Peninsula "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus"
# 

# ![image](images/iris.png)

# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.

# ## A. Basic - Iris flower data set 

# ### Exercise 1.

# Write a Python program to load the iris data from a given csv file into a dataframe and print the shape of the data, type of the data and first 3 rows.

# ```files/exercise_A.1.py```

# In[145]:


import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("iris.csv")
print("Shape of the data:")
print(data.shape)
print("\nData Type:")
print(type(data))
print("\nFirst 3 rows:")
print(data.head(3))


# ### Exercise 2. 

# Write a Python program using Scikit-learn to print the keys, number of rows-columns, feature names and the description of the Iris data. 

# ```files/exercise_A.2.py```

# In[35]:


print("\nKeys of Iris dataset:")
print(data.keys())
print("\nNumber of rows and columns of Iris dataset:")
print(data.shape) 


# ### Exercise 3. 

# Write a Python program to get the number of observations, missing values and nan values.

# ```files/exercise_A.3.py```

# In[36]:


print(data.info())


# ### Exercise 4.

# Write a Python program to create a 2-D array with ones on the diagonal and zeros elsewhere. Now convert the NumPy array to a SciPy sparse matrix in CSR format. 
# 
# From Wikipedia:
# 
# ```In numerical analysis and scientific computing, a sparse matrix or sparse array is a matrix in which most of the elements are zero. By contrast, if most of the elements are nonzero, then the matrix is considered dense. The number of zero-valued elements divided by the total number of elements (e.g., m x n for an m x n matrix) is called the sparsity of the matrix (which is equal to 1 minus the density of the matrix). Using those definitions, a matrix will be sparse when its sparsity is greater than 0.5.```

# ```files/exercise_A.4.py```

# In[8]:


eye = np.eye(4)
print("NumPy array:\n", eye)
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n", sparse_matrix)


# ### Exercise 5.

# Write a Python program to view basic statistical details like percentile, mean, std etc. of iris data.

# ```files/exercise_A.5.py```

# In[9]:


print(data.describe())


# ### Exercise 6. 

# Write a Python program to view basic statistical details like percentile, mean, std etc. of iris data.

# ```files/exercise_A.6.py```

# In[11]:


print("Observations of each species:")
print(data['Name'].value_counts()) 


# ### Exercise 7. 

# Write a Python program to drop Id column from a given Dataframe and print the modified part. Call iris.csv to create the Dataframe.

# ```files/exercise_A.7.py```

# In[64]:


#Had to set the first col as index to remove it
data = pd.read_csv("iris.csv")
print("Original Data:")
print(data.head())
data.set_index('SepalLength', inplace=True)
print("After removing id column:")
print(data.head()) 


# ### Exercise 8.

# Write a Python program to access first four cells from a given Dataframe using the index and column labels. Call iris.csv to create the Dataframe.

# ```files/exercise_A.8.py```

# In[85]:


data = pd.read_csv("iris.csv")
# data.set_index('SepalLength', inplace=True)
x = [data.iloc[0][0], data.iloc[0][1], data.iloc[0][2], data.iloc[0][3]]
print(x) 


# ## B. Visualization - Iris flower data set

# ### Exercise 1.

# Write a Python program to create a plot to get a general Statistics of Iris data.

# ```files/exercise_B.1.py```

# In[87]:


import pandas as pd
import matplotlib.pyplot as plt
iris = pd.read_csv("iris.csv")
iris.describe().plot(kind = "area",fontsize=16, figsize = (15,8), table = True, colormap="Accent")
plt.xlabel('Statistics',)
plt.ylabel('Value')
plt.title("General Statistics of Iris Dataset")
plt.show()


# ### Exercise 2. 

# Write a Python program to create a Bar plot to get the frequency of the three species of the Iris data.

# ```files/exercise_B.2.py```

# In[91]:


iris = pd.read_csv("iris.csv")
ax=plt.subplots(1,1,figsize=(10,8))
sns.countplot('Name',data=iris)
plt.title("Iris Species Count")
plt.show()


# ### Exercise 3. 

# Write a Python program to create a Pie plot to get the frequency of the three species of the Iris data.

# ```files/exercise_B.3.py```

# In[93]:


iris = pd.read_csv("iris.csv")
ax=plt.subplots(1,1,figsize=(10,8))
iris['Name'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',shadow=True,figsize=(10,8))
plt.title("Iris Species %")
plt.show()


# ### Exercise 4. 

#  Write a Python program to create a graph to find relationship between the sepal length and width.

# ```files/exercise_B.4.py```

# In[96]:


iris = pd.read_csv("iris.csv")
fig = iris[iris.Name=='Iris-setosa'].plot(kind='scatter',x='SepalLength',y='SepalWidth',color='orange', label='Setosa')
iris[iris.Name=='Iris-versicolor'].plot(kind='scatter',x='SepalLength',y='SepalWidth',color='blue', label='versicolor',ax=fig)
iris[iris.Name=='Iris-virginica'].plot(kind='scatter',x='SepalLength',y='SepalWidth',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()


# ### Exercise 5.

# Write a Python program to create a graph to find relationship between the petal length and width.

# ```files/exercise_B.5.py```

# In[98]:


iris = pd.read_csv("iris.csv")
fig = iris[iris.Name=='Iris-setosa'].plot.scatter(x='PetalLength',y='PetalWidth',color='orange', label='Setosa')
iris[iris.Name=='Iris-versicolor'].plot.scatter(x='PetalLength',y='PetalWidth',color='blue', label='versicolor',ax=fig)
iris[iris.Name=='Iris-virginica'].plot.scatter(x='PetalLength',y='PetalWidth',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()


# ### Exercise 6.

# Write a Python program to create a graph to see how the length and width of SepalLength, SepalWidth, PetalLength, PetalWidth are distributed.

# ```files/exercise_B.6.py```

# In[100]:


iris = pd.read_csv("iris.csv")

iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,12)
plt.show()


# ### Exercise 7.

# Write a Python program to create a joinplot to describe individual distributions on the same plot between Sepal length and Sepal width. Note: joinplot - Draw a plot of two variables with bivariate and univariate graphs.

# ```files/exercise_B.7.py```

# In[101]:


iris = pd.read_csv("iris.csv")
fig=sns.jointplot(x='SepalLength', y='SepalWidth', data=iris, color='blue') 
plt.show()


# ### Exercise 8. 

# Write a Python program to create a joinplot using "hexbin" to describe individual distributions on the same plot between Sepal length and Sepal width.  Note: The bivariate analogue of a histogram is known as a "hexbin" plot, because it shows the counts of observations that fall within hexagonal bins. This plot works best with relatively large datasets. It's available through the matplotlib plt.hexbin function and as a style in jointplot(). It looks best with a white background.

# ```files/exercise_B.8.py```

# In[102]:


iris = pd.read_csv("iris.csv")
fig=sns.jointplot(x='SepalLength', y='SepalWidth', kind="hex", color="red", data=iris)
plt.show()


# ### Exercise 9. 

# Write a Python program to create a joinplot using "kde" to describe individual distributions on the same plot between Sepal length and Sepal width. Note:  The kernel density estimation (kde) procedure visualize a bivariate distribution. In seaborn, this kind of plot is shown with a contour plot and is available as a style in jointplot().

# ```files/exercise_B.9.py```

# In[103]:


iris = pd.read_csv("iris.csv")
fig=sns.jointplot(x='SepalLength', y='SepalWidth', kind="kde", color='cyan', data=iris)  
plt.show()


# ### Exercise 10. 

# Write a Python program to create a joinplot and add regression and kernel density fits using "reg" to describe individual distributions on the same plot between Sepal length and Sepal width.

# ```files/exercise_B.10.py```

# In[104]:


iris = pd.read_csv("iris.csv")
fig=sns.jointplot(x='SepalLength', y='SepalWidth', kind="reg", color='red', data=iris) 
plt.show()


# ### Exercise 11.

# Write a Python program to draw a scatterplot, then add a joint density estimate to describe individual distributions on the same plot between Sepal length and Sepal width.

# ```files/exercise_B.11.py```

# In[105]:


iris = pd.read_csv("iris.csv")
sns.jointplot("SepalLength", "SepalWidth", data=iris, color="b").plot_joint(sns.kdeplot, zorder=0, n_levels=6) 
plt.show()


# ### Exercise 12.

# Write a Python program to create a joinplot using "kde" to describe individual distributions on the same plot between Sepal length and Sepal width and use '+' sign as marker.  Note:  The kernel density estimation (kde) procedure visualize a bivariate distribution. In seaborn, this kind of plot is shown with a contour plot and is available as a style in jointplot().

# ```files/exercise_B.12.py```

# In[106]:


iris = pd.read_csv("iris.csv")
g = sns.jointplot(x="SepalLength", y="SepalWidth", data=iris, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=40, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$SepalLength()$", "$SepalWidth()$") 
plt.show()


# ### Exercise 13. 

# Write a Python program to create a pairplot of the iris data set and check which flower species seems to be the most separable.

# ```files/exercise_B.13.py```

# In[107]:


iris = pd.read_csv("iris.csv")
g = sns.jointplot(x="SepalLength", y="SepalWidth", data=iris, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=40, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$SepalLength()$", "$SepalWidth()$") 
plt.show()


# ### Exercise 14.

#  Write a Python program using seaborn to Create a kde (Kernel Density Estimate ) plot of sepal_length versus sepal width for setosa species of flower.

# ```files/exercise_B.14.py```

# In[108]:


iris = pd.read_csv("iris.csv")
sub=iris[iris['Name']=='Iris-setosa']
sns.kdeplot(data=sub[['SepalLength','SepalWidth']],ap="plasma", shade=True, shade_lowest=False)
plt.title('Iris-setosa')
plt.xlabel('Sepal Length ')
plt.ylabel('Sepal Width ')
plt.show()


# ### Exercise 15.

# Write a Python program using seaborn to Create a kde (Kernel Density Estimate ) plot of petal_length versus petal width for setosa species of flower.

# ```files/exercise_B.15.py```

# In[109]:


iris = pd.read_csv("iris.csv")
sns.kdeplot(data=sub[['PetalLength','PetalWidth']],ap="plasma", shade=True, shade_lowest=False)
plt.title('Iris-setosa')
plt.xlabel('Petal Length ')
plt.ylabel('Petal Width ')
plt.show()


# ### Exercise 16.

# Write a Python program using seaborn to Create a kde (Kernel Density Estimate ) plot of petal_length versus petal width for setosa species of flower.

# ```files/exercise_B.16.py```

# In[110]:


iris = pd.read_csv("iris.csv")
sns.kdeplot(data=sub[['PetalLength','PetalWidth']],ap="plasma", shade=True, shade_lowest=False)
plt.title('Iris-setosa')
plt.xlabel('Petal Length ')
plt.ylabel('Petal Width ')
plt.show()


# ### Exercise 17.

# Write a Python program to find the correlation between variables of iris data. Also create a hitmap using Seaborn to present their relations.

# ```files/exercise_B.17.py```

# In[114]:


iris = pd.read_csv("iris.csv")
X = iris.iloc[:, 0:4]
f, ax = plt.subplots(figsize=(10, 8))
corr = X.corr()
print(corr)
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), square=True, ax=ax, linewidths=.5)
plt.show() 


# ### Exercise 18.

# Write a Python program to create a box plot (or box-and-whisker plot) which shows the distribution of quantitative data in a way that facilitates comparisons between variables or across levels of a categorical variable of iris dataset. Use seaborn. 

# ```files/exercise_B.18.py```

# In[115]:


iris = pd.read_csv("iris.csv")

box_data = iris #variable representing the data array
box_target = iris.Name #variable representing the labels array
sns.boxplot(data = box_data,width=0.5,fliersize=5)
sns.set(rc={'figure.figsize':(2,15)})


# ### Exercise 19.

# From Wikipedia -
# 
# ```Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components. The resulting vectors (each being a linear combination of the variables and containing n observations) are an uncorrelated orthogonal basis set. PCA is sensitive to the relative scaling of the original variables.```  
# 
# Write a Python program to create a Principal component analysis (PCA) of iris dataset.

# ```files/exercise_B.19.py```

# In[128]:


iris = pd.read_csv("iris.csv")
# Converting string labels into numbers.
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
iris.Name = le.fit_transform(iris.Name)
#Drop id column
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values

fig = plt.figure(1, figsize=(7, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)
for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
plt.show()


# ## C. K-Nearest Neighbors Algorithm in Iris flower data set

# From Wikipedia,
# 
# ```In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression.[1] In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:```
# 
# - itemscope itemtype="http://schema.org/WebPageElement/Heading"> In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
# - itemscope itemtype="http://schema.org/WebPageElement/Heading"> In k-NN regression, the output is the property value for the object. This value is the average of the values of k nearest neighbors.

# ![image](images/neighbors.png)

# Example of k-NN classification. The test sample (green dot) should be classified either to blue squares or to red triangles. If k = 3 (solid line circle) it is assigned to the red triangles because there are 2 triangles and only 1 square inside the inner circle. If k = 5 (dashed line circle) it is assigned to the blue squares (3 squares vs. 2 triangles inside the outer circle).

# ### Exercise 1.

# Write a Python program to split the iris dataset into its attributes (X) and labels (y). The X variable contains the first four columns (i.e. attributes) and y contains the labels of the dataset.

# ```files/exercise_C.1.py```

# In[130]:


iris = pd.read_csv("iris.csv")

X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
print("Attributes:")
print(X)
print("\nLabels:")
print(y)


# ### Exercise 2.

# Write a Python program using Scikit-learn to split the iris dataset into 70% train data and 30% test data. Out of total 150 records, the training set will contain 120 records and the test set contains 30 of those records. Print both datasets.

# ```files/exercise_C.2.py```

# In[132]:


iris = pd.read_csv("iris.csv")
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
#Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print("\n70% train data:")
print(X_train)
print(y_train)
print("\n30% test data:")
print(X_test)
print(y_test)


# ### Exercise 3.

# Write a Python program using Scikit-learn to convert Species columns in a numerical column of the iris dataframe. To encode this data map convert each value to a number. e.g. Iris-setosa:0, Iris-versicolor:1, and Iris-virginica:2. Now print the iris dataset into 80% train data and 20% test data. Out of total 150 records, the training set will contain 120 records and the test set contains 30 of those records. Print both datasets.

# ```files/exercise_C.3.py```

# In[134]:


iris = pd.read_csv("iris.csv")
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
iris.Name = le.fit_transform(iris.Name)
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
#Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print("\n80% train data:")
print(X_train)
print(y_train)
print("\n20% test data:")
print(X_test)
print(y_test)


# ### Exercise 4.

# Write a Python program using Scikit-learn to split the iris dataset into 70% train data and 30% test data. Out of total 150 records, the training set will contain 105 records and the test set contains 45 of those records. Predict the response for test dataset (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm) using the K Nearest Neighbor Algorithm. Use 5 as number of neighbors.

# ```files/exercise_C.4.py```

# In[136]:


iris = pd.read_csv("iris.csv")
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
#Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
'''
print("\n70% train data:")
print(X_train)
print(y_train)
print("\n30% test data:")
print(X_test)
print(y_test)
'''
#Create KNN Classifier
#Number of neighbors to use by default for kneighbors queries.
knn = KNeighborsClassifier(n_neighbors=5)
#Train the model using the training sets
knn.fit(X_train, y_train)
#Predict the response for test dataset
print("Response for test dataset:")
y_pred = knn.predict(X_test)
print(y_pred)


# ### Exercise 5.

# Write a Python program using Scikit-learn to split the iris dataset into 80% train data and 20% test data. Out of total 150 records, the training set will contain 120 records and the test set contains 30 of those records. Train or fit the data into the model and calculate the accuracy of the model using the K Nearest Neighbor Algorithm.

# ```files/exercise_C.5.py```

# In[137]:


iris = pd.read_csv("iris.csv")
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
#Split arrays or matrices into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
knn = KNeighborsClassifier(n_neighbors=7)  
knn.fit(X_train, y_train)   
# Calculate the accuracy of the model 
print("Accuracy of the model:")
print(knn.score(X_test, y_test))


# ### Exercise 6.

# Write a Python program using Scikit-learn to split the iris dataset into 80% train data and 20% test data. Out of total 150 records, the training set will contain 120 records and the test set contains 30 of those records. Train or fit the data into the model and using the K Nearest Neighbor Algorithm calculate the performance for different values of k.

# ```files/exercise_C.6.py```

# In[138]:


iris = pd.read_csv("iris.csv")
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
#Split arrays or matrices into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
knn = KNeighborsClassifier(n_neighbors=7)  
knn.fit(X_train, y_train)   
# Calculate the accuracy of the model for different values of k
for i in np.arange(1, 10):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X_train, y_train)
    print("For k = %d accuracy is"%i,knn2.score(X_test,y_test))


# ### Exercise 7.

# Write a Python program using Scikit-learn to split the iris dataset into 80% train data and 20% test data. Out of total 150 records, the training set will contain 120 records and the test set contains 30 of those records. Train or fit the data into the model and using the K Nearest Neighbor Algorithm and create a plot to present the performance for different values of k.

# ```files/exercise_C.7.py```

# In[140]:


iris = pd.read_csv("iris.csv")
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
#Split arrays or matrices into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
knn = KNeighborsClassifier(n_neighbors=7)  
knn.fit(X_train, y_train)   
a_index=list(range(1,11))
a=pd.Series()
# Calculate the accuracy of the model for different values of k
for i in np.arange(1, 10):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X_train, y_train)
    print("For k = %d accuracy is"%i,knn2.score(X_test,y_test))
# Visual presentation: Various values of n for K-Nearest nerighbours
print("\nVisual presentation: Various values of n for K-Nearest nerighbours:")    
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,y_test)))
plt.plot(a_index, a)


# ### Exercise 8.

# Write a Python program using Scikit-learn to split the iris dataset into 80% train data and 20% test data. Out of total 150 records, the training set will contain 120 records and the test set contains 30 of those records. Train or fit the data into the model and using the K Nearest Neighbor Algorithm and create a plot of k values vs accuracy.

# ```files/exercise_C.8.py```

# In[141]:


iris = pd.read_csv("iris.csv")
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
#Split arrays or matrices into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
knn = KNeighborsClassifier(n_neighbors=7)  
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)

print("Preliminary model score:")
print(knn.score(X_test,y_test))

no_neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(no_neighbors))
test_accuracy = np.empty(len(no_neighbors))

for i, k in enumerate(no_neighbors):
    # We instantiate the classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Visualization of k values vs accuracy

plt.title('k-NN: Varying Number of Neighbors')
plt.plot(no_neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(no_neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# ## D. Logistic Regression in Sci-Kit Learn

# ### Exercise 1.

# Write a Python program to view some basic statistical details like percentile, mean, std etc. of the species of 'Iris-setosa', 'Iris-versicolor' and 'Iris-versicolor'.

# ```files/exercise_D.1.py```

# In[142]:


data = pd.read_csv("iris.csv")
print('Iris-setosa')
setosa = data['Name'] == 'Iris-setosa'
print(data[setosa].describe())
print('\nIris-versicolor')
setosa = data['Name'] == 'Iris-versicolor'
print(data[setosa].describe())
print('\nIris-virginica')
setosa = data['Name'] == 'Iris-virginica'
print(data[setosa].describe())


# ### Exercise 2.

# Write a Python program to create a scatter plot using sepal length and petal_width to separate the Species classes.

# ```files/exercise_D.2.py```

# In[144]:


iris = pd.read_csv("iris.csv")
#Convert Name columns in a numerical column of the iris dataframe
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
iris.Name = le.fit_transform(iris.Name)
x = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
plt.scatter(x[:,0], x[:, 3], c=y)
plt.xlabel('Sepal Length ')
plt.ylabel('Petal Width ')
plt.show()


# ### Exercise 3.

# In statistical modeling, regression analysis is a set of statistical processes for estimating the relationships among variables. It includes many techniques for modeling and analyzing several variables, when the focus is on the relationship between a dependent variable and one or more independent variables (or 'predictors'). Write a Python program to get the accuracy of the Logistic Regression.

# ```files/exercise_D.3.py```

# In[146]:


iris = pd.read_csv("iris.csv")
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values

#Split arrays or matrices into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 

model = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, y)
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('The accuracy of the Logistic Regression is', metrics.accuracy_score(prediction,y_test))


# In[ ]:




