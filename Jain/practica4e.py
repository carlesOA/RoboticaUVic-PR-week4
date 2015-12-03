import numpy as np
from sklearn.svm import LinearSVC, SVC

print "Solving Q5\n"

data = np.loadtxt('jain.txt')

sel = np.random.permutation(data.shape[0])

all_data = data[:,:-1]
data_lab = data[:,-1] 	

l = (len(data)/2) 

train = all_data[sel[:l],:]
test = all_data[sel[:l],:]

train_lab = data_lab[sel[:l]]
test_lab = data_lab[sel[:l]]
		
clf = LinearSVC(C=5)
clf.fit(train, train_lab)
print 'Linear SVM score', clf.score(test, test_lab)


clfnl = SVC(C=5)
clfnl.fit(train, train_lab)
print 'RBF SVM score', clfnl.score(test, test_lab)

def paint_decision_functions(data, labels, clf):  
    from matplotlib.colors import ListedColormap  
    import pylab  
    cm = pylab.cm.RdBu  
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])  
    x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5  
    y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5  
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),  
                         np.arange(y_min, y_max, 0.1))  
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  
    Z = Z.reshape(xx.shape)  
    pylab.contourf(xx, yy, Z, cmap=cm, alpha=.8)  
    pylab.scatter(data[:, 0], data[:, 1], c=labels, cmap=cm_bright)  
    pylab.xlim(xx.min(), xx.max())  
    pylab.ylim(yy.min(), yy.max())  
    pylab.xticks(())  
    pylab.yticks(())  
    pylab.show()  

#paint_decision_functions(train, train_lab, clf)
paint_decision_functions(test, test_lab, clf)
