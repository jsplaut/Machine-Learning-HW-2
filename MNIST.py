import matplotlib.pyplot as plt
import numpy as np

from read_mnist import load_data, pretty_print
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection, neighbors, metrics
from sklearn.model_selection import cross_val_score

def logisticRegression1(xd):
    train_set, test_set = load_data()

    yTrain = train_set[1]
    xTrain = train_set[0]
    yTest = test_set[1]
    xTest = test_set[0]

    # scaler = StandardScaler()
    # xTrain = scaler.fit_transform(xTrain)
    # xTest = scaler.transform(xTest)

    train_samples = 5000
    clf = LogisticRegression(C=1.0,
                            multi_class='multinomial',
                            penalty='l1', solver='saga', tol=0.1)

    clf.fit(xTrain, yTrain)
    sparsity = np.mean(clf.coef_ == 0) * 100
    score = clf.score(xTest, yTest)
    pred = clf.predict(xTest)
    # print('Best C % .4f' % clf.C_)

    print("Sparsity with L1 penalty: %.2f%%" % sparsity)
    print("Test score with L1 penalty: %.4f" % score)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
        clf.score(xTest, yTest)))
    print(metrics.classification_report(yTest, pred, digits = 4))

    coef = clf.coef_.copy()
    plt.figure(figsize=(10, 5))
    scale = np.abs(coef).max()
    for i in range(10):
        l1_plot = plt.subplot(2, 5, i + 1)
        l1_plot.imshow(coef[i].reshape(28, 28), interpolation='nearest',
                       cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
        l1_plot.set_xticks(())
        l1_plot.set_yticks(())
        l1_plot.set_xlabel('Class %i' % i)
    plt.suptitle('Classification vector for...')

    #plt.show()

def logiCrossValidation():
    train_set, test_set = load_data()

    yTrain = train_set[1]
    xTrain = train_set[0]

    kfold = model_selection.KFold(n_splits=10, random_state=7)
    modelCV = LogisticRegression()
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, xTrain, yTrain, cv=kfold, scoring=scoring)
    print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

def baseLines(xd):
    train_set, test_set = load_data()

    yTest=test_set[1]

    size = len(yTest)

    for x in range(0,10):
        count = 0.0
        for z in range(0, size):
            if x == yTest[z]:
                count+=1.0
        acc = (count/size)*100
        print("The accuracy of %d: %.2f " % (x, acc))

def knnDistance(xd):
    train_set, test_set = load_data()

    yTrain = train_set[1]
    xTrain = train_set[0]
    yTest = test_set[1]
    xTest = test_set[0]

    # scaler = StandardScaler()
    # xTrain = scaler.fit_transform(xTrain)
    # xTest = scaler.transform(xTest)
    
    clf = neighbors.KNeighborsClassifier(weights = 'distance')
    clf.fit(xTrain,yTrain)
    pred = clf.predict(xTest)

    score = clf.score(xTest, yTest)

    matrix = metrics.confusion_matrix(yTest, pred)

    print(metrics.classification_report(yTest, pred))
    print(matrix) 
    print("Test score with L1 penalty: %.4f" % score)
    
def knn(xD):
    train_set, test_set = load_data()

    yTrain = train_set[1]
    xTrain = train_set[0]
    yTest = test_set[1]
    xTest = test_set[0]

    nneighbors = [3,5,7,9,11]
    for n in nneighbors:
        clf = neighbors.KNeighborsClassifier(n_neighbors = n)
        clf.fit(xTrain, yTrain)
        pred = clf.predict(xTest)
        score = clf.score(xTest, yTest)
        print("\n Accuracy Score for %d neighbors: %.4f \n" % (n, score))
        print(metrics.classification_report(yTest,pred, digits = 4))



#logisticRegression1(1)
#baseLines(1)
#logiCrossValidation()
#knnDistance(1)
knn(1)
