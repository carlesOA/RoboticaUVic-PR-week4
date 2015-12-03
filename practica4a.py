import numpy as np
import scipy.spatial.distance as dist
import csv
import math
from sklearn.metrics import accuracy_score as accuracy_score
from collections import Counter

print "Solving Q1"


data = np.array([map(float, x.split(',')[:-1]) for x in open('iris.data') if x.strip()!=''])
labels = np.array([x.split(',')[-1].strip() for x in open('iris.data') if x.strip()!=''])

idx_train = np.loadtxt('iris_idx_train.txt')
idx_test = np.loadtxt('iris_idx_test.txt')

idx_train = idx_train.astype(int)
idx_test = idx_test.astype(int)

data_train = data[idx_train,:]
data_test = data[idx_test,:]
labels_train = labels[idx_train]
labels_test = labels[idx_test]

def kNN(data_train, data_test, labels_train, labels_test):
    for n in range(1,9,1):	
	distance = dist.cdist(data_train, data_test)
	min_k = np.argsort(distance.T,1)[:,1:n+1]
	min_labels = labels_train[min_k]	
	acc = accuracy_score(labels_test, [Counter(x).most_common()[0][0] for x in min_labels] )
	print acc
	

kNN (data_train, data_test, labels_train, labels_test)
