# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 12:24:58 2021

@author: justi
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd
import re 
import matplotlib
import gzip
import os
import numpy as np
import csv
data1 = pd.read_csv('C:/Users/justi/OneDrive/Documents/Spring 2021/ASTR 6410/Final Project PCA/Genome doc_NV_NR.csv', header=None)
adata =np.array(data1)
data =[]
for i in range(0,6):#this value may need to change (col. number) based on how many variables
    data.append(adata[:,i])
print(data)

def standardize_data(arr):
         
    '''
    This function standardize an array, its substracts mean value, 
    and then divide the standard deviation.
    
    param 1: array 
    return: standardized array
    '''    
    rows, columns = arr.shape
    
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    
    for column in range(columns):
        
        mean = np.mean(arr[:,column].astype(int))
        std = np.std(arr[:,column].astype(int))
        tempArray = np.empty(0)
        
        for element in arr[:,column].astype(int):
            
            tempArray = np.append(tempArray, ((element - mean) / std))
 
        standardizedArray[:,column] = tempArray
    
    return standardizedArray
arr = data1.iloc[1:,0:4].values #this value and the next may need to change (col. numbers) based on how many variables
Y = data1.iloc[1:,5].values
X = standardize_data(arr)
# Calculating the covariance matrix
covariance_matrix = np.cov(X.T)

eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
print("Eigenvector: \n",eigen_vectors,"\n")
print("Eigenvalues: \n", eigen_values, "\n")

# Calculating the explained variance on each of components
variance_explained = []
for i in eigen_values:
     variance_explained.append((i/sum(eigen_values))*100)
        
print("Variance by position", variance_explained)
# Identifying components that explain at least 95%
cumulative_variance_explained = np.cumsum(variance_explained)
print("Cumulative variance by position", cumulative_variance_explained)

# Visualizing the eigenvalues and finding the "elbow" in the graphic
fig = plt.figure()
sns.lineplot(x = [1,2,3,4], y=cumulative_variance_explained)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Explained variance vs Number of components")
# Visualizing the eigenvalues and finding the "elbow" in the graphic
# fig = plt.figure()
# sns.lineplot(x = [1,2,3,4,5], y=variance_explained)
# plt.xlabel("Number of components")
# plt.ylabel("Cumulative explained variance")
# plt.title("Explained variance vs Number of components")

# Using component 1 and 4 (which I moved to be 1 and 2)
projection_matrix = (eigen_vectors.T[:3][:2]).T #indicate desired column after .T in [:#colnumber]
print(projection_matrix)

# Getting the product of original standardized X and the eigenvectors 
X_pca = X.dot(projection_matrix)
print(X_pca)
#plot points as PCA1 vs. PCA2 for grouping

fig = plt.figure()
for i in range(1,8):
    plt.scatter(X_pca[i:,0], X_pca[i:,1], color = "red")
for i in range(9,11):
    plt.scatter(X_pca[i:,0], X_pca[i:,1], color = "blue")
for i in range(12,22):
    plt.scatter(X_pca[i:,0], X_pca[i:,1], color ="lime")
for i in range(22,32):
    plt.scatter(X_pca[i:,0], X_pca[i:,1], color = "black")
for i in range(33,39):
    plt.scatter(X_pca[i:,0], X_pca[i:,1], color = "darkviolet")
for i in range(40,48):
    plt.scatter(X_pca[i:,0], X_pca[i:,1], color = "cyan")
plt.xlabel("PCA 1st Component [59.55%]")
plt.ylabel("PCA 2nd Component[25.37%]")