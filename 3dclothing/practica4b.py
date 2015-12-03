import numpy as np
import math
from sklearn.linear_model import LogisticRegression
import pylab as pl

print "Solving Q2"
D = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
p = 0
data_train = np.load('3dclothing_train.npy')
data_test = np.load('3dclothing_test.npy')

labels_train = np.array([x.strip() for x in open('3dclothing_labels_train.txt')])
labels_test = np.array([x.strip() for x in open('3dclothing_labels_test.txt')])

ok_test=data_test[(labels_test == 'shirt') + (labels_test == 'polo shirt'),:]
ok_train=data_train[(labels_train == 'shirt') + (labels_train == 'polo shirt'),:]

ok_lab_train = labels_train[(labels_train == 'shirt') + (labels_train == 'polo shirt')]
ok_lab_test = labels_test[(labels_test == 'shirt') + (labels_test == 'polo shirt')]

for i in xrange (-7,7):    
	LogReg = LogisticRegression(C=10**i)
	D[p]=i
	LogReg.fit(ok_train, ok_lab_train)
	a[p] = LogReg.score(ok_test, ok_lab_test)
	print 'Log Reg accuracy', a[p]
	p+=1

pl.plot(D[:14],a[:14])

pl.show()
