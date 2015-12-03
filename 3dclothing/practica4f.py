import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

print "Solving Q6\n"

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
	LogReg.fit(ok_train, ok_lab_train)
	a = LogReg.score(ok_test, ok_lab_test)
	print 'Log Reg accuracy', a
	clf = SVC(C=10**i)
	clf.fit(data_train, labels_train)
	print 'RBF SVM accuracy', clf.score(data_test, labels_test)




