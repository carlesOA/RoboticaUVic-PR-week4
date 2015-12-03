import numpy as np
import math
from sklearn.linear_model import LogisticRegression


print "Solving Q4"

data_train = np.load('3dclothing_train.npy')
data_test = np.load('3dclothing_test.npy')

labels_train = np.array([x.strip() for x in open('3dclothing_labels_train.txt')])
labels_test = np.array([x.strip() for x in open('3dclothing_labels_test.txt')])

ok_test=data_test[(labels_test == 'shirt') + (labels_test == 'jeans'),:]
ok_train=data_train[(labels_train == 'shirt') + (labels_train == 'jeans'),:]

ok_lab_train = labels_train[(labels_train == 'shirt') + (labels_train == 'jeans')]
ok_lab_test = labels_test[(labels_test == 'shirt') + (labels_test == 'jeans')]

for i in xrange (-7,7):    
	LogReg = LogisticRegression(C=10**i)
	LogReg.fit(ok_train, ok_lab_train)
	a = LogReg.score(ok_test, ok_lab_test)
	bias = LogReg.intercept_
	theta = LogReg.coef_
	acc = 1/(1+(np.exp(-np.dot(ok_test, theta.T)-bias)))
	print 'New', acc
