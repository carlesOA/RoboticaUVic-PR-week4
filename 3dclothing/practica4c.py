import numpy as np
import math
from sklearn.linear_model import LogisticRegression
import pylab as pl

print "Solving Q3"
D = [1,2,3,4,5,6]
a = [1,2,3,4,5,6]
p = 0
data_train = np.load('3dclothing_train.npy')
data_test = np.load('3dclothing_test.npy')

labels_train = np.array([x.strip() for x in open('3dclothing_labels_train.txt')])
labels_test = np.array([x.strip() for x in open('3dclothing_labels_test.txt')])

for i in xrange (-3,3):    
	LogReg = LogisticRegression(C=10**i)
	D[p]=i
	LogReg.fit(data_train, labels_train)
	a[p] = LogReg.score(data_test, labels_test)
	print 'Log Reg accuracy', a[p]
	p+=1

pl.plot(D[:5],a[:5])

pl.show()
