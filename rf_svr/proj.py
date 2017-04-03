import pandas as pd
import numpy as np
from time import sleep
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


#Loading the data set from competition
train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

#printing the dataset 
print("Training set has {0[0]} rows and {0[1]} columns".format(train_set.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test_set.shape))

train_set = train_set.as_matrix()
test_set = test_set.as_matrix()

print len(train_set)
idx = np.random.choice(np.arange(len(train_set)), 1000, replace=False)
X = train_set[idx,1:]
y = train_set[idx,0]
ss = StandardScaler()
ss.fit(X)
X = ss.transform(X)

X_test = test_set
X_test = ss.transform(X_test)



#print("\nRandom Forest Classifier 2 estimators")
#tr_perf=[]
#val_perf=[]
##10-fold cross validation on data
#kf = KFold(n_splits=10)
#fold=0;
#for train_index, test_index in kf.split(X):
#  X_train, X_val = X[train_index], X[test_index]
#  y_train, y_val = y[train_index], y[test_index]
#
#  #Random Forest algorithm
#  rfc = RandomForestClassifier(n_estimators = 2)
#  rfc.fit(X_train, y_train)
#  tr_perf.append(rfc.score(X_train, y_train))
#  print("Accuracy on training set fold: {:.3f}".format(tr_perf[fold]))
#  val_perf.append(accuracy_score(rfc.predict(X_val),y_val))
#  print("Accuracy on validation set fold: {:.3f}".format(val_perf[fold]))
#  fold += 1
#print("Average accuracy on training set fold: {:.3f} +/-{:.3f}".format(np.mean(tr_perf),np.std(tr_perf)))
#print("Average accuracy on validation set fold: {:.3f} +/-{:.3f}".format(np.mean(val_perf),np.std(val_perf)))
#
##Random Forest algorithm
#rfc = RandomForestClassifier(n_estimators = 2)
#rfc.fit(X_train, y_train)
#print("Accuracy on full training set: {:.3f}".format(rfc.score(X_train, y_train)))
#result = rfc.predict(X_test)
##To save results into .csv file and submit to kaggle
#df = pd.DataFrame(result)
#df.index.name='ImageId'
#df.index+=1
#df.columns=['Label']
#df.to_csv('result-rf-2.csv', header=True)
#
#
#print("\nRandom Forest Classifier 10 estimators")
#tr_perf=[]
#val_perf=[]
##10-fold cross validation on data
#kf = KFold(n_splits=10)
#fold=0;
#for train_index, test_index in kf.split(X):
#  X_train, X_val = X[train_index], X[test_index]
#  y_train, y_val = y[train_index], y[test_index]
#
#  #Random Forest algorithm
#  rfc = RandomForestClassifier(n_estimators = 10)
#  rfc.fit(X_train, y_train)
#  tr_perf.append(rfc.score(X_train, y_train))
#  print("Accuracy on training set fold: {:.3f}".format(tr_perf[fold]))
#  val_perf.append(accuracy_score(rfc.predict(X_val),y_val))
#  print("Accuracy on validation set fold: {:.3f}".format(val_perf[fold]))
#  fold += 1
#print("Average accuracy on training set fold: {:.3f} +/-{:.3f}".format(np.mean(tr_perf),np.std(tr_perf)))
#print("Average accuracy on validation set fold: {:.3f} +/-{:.3f}".format(np.mean(val_perf),np.std(val_perf)))
#
##Random Forest algorithm
#rfc = RandomForestClassifier(n_estimators = 10)
#rfc.fit(X_train, y_train)
#print("Accuracy on full training set: {:.3f}".format(rfc.score(X_train, y_train)))
#result = rfc.predict(X_test)
##To save results into .csv file and submit to kaggle
#df = pd.DataFrame(result)
#df.index.name='ImageId'
#df.index+=1
#df.columns=['Label']
#df.to_csv('result-rf-10.csv', header=True)
#
#
#
##Support vector classifier
#print("\nSupport vector classifier all dimensions")
##grid search for best set of hyper parameters
#parameters = {'kernel':['rbf'], 'C':np.logspace(-5,15,21,base=2),
#'gamma':np.logspace(-15,3,19,base=2), 'epsilon':np.linspace(0,0.5,11)}
#svr = SVR()
#clf = GridSearchCV(svr, parameters, n_jobs=-1, verbose=2, cv=3)
#clf.fit(X, y)
#best = clf.best_params_
#print("Best parameters are:")
#print(best)
##past best were: 
##best = {'epsilon': 0.0, 'C': 32.0, 'gamma': 0.001953125}
#
#tr_perf=[]
#val_perf=[]
##10-fold cross validation on data
#kf = KFold(n_splits=10)
#fold=0;
#for train_index, test_index in kf.split(X):
#  X_train, X_val = X[train_index], X[test_index]
#  y_train, y_val = y[train_index], y[test_index]
#
#  #svr
#  svr = SVR(kernel='rbf', epsilon=best['epsilon'], C=best['C'], gamma=best['gamma'])
#  svr.fit(X_train, y_train)
#  pred = np.around(svr.predict(X_train)).astype(int)
#  pred[pred<0]=0
#  pred[pred>9]=9
#  tr_perf.append(accuracy_score(pred, y_train))
#  print("Accuracy on training set fold: {:.3f}".format(tr_perf[fold]))
#  pred = np.around(svr.predict(X_val)).astype(int)
#  pred[pred<0]=0
#  pred[pred>9]=9
#  val_perf.append(accuracy_score(pred,y_val))
#  print("Accuracy on validation set fold: {:.3f}".format(val_perf[fold]))
#  fold += 1
#print("Average accuracy on training set fold: {:.3f} +/-{:.3f}".format(np.mean(tr_perf),np.std(tr_perf)))
#print("Average accuracy on validation set fold: {:.3f} +/-{:.3f}".format(np.mean(val_perf),np.std(val_perf)))
#
##final svr algorithm
#svr = SVR(kernel='rbf', epsilon=best['epsilon'], C=best['C'], max_iter=-1, verbose=True)
#svr.fit(X_train, y_train)
#pred = np.around(svr.predict(X_train)).astype(int)
#pred[pred<0]=0
#pred[pred>9]=9
#print("Accuracy on full training set: {:.3f}".format(accuracy_score(pred, y_train)))
#result = np.around(svr.predict(X_test)).astype(int)
#result[result<0]=0
#result[result>9]=9
##To save results into .csv file and submit to kaggle
#df = pd.DataFrame(result)
#df.index.name='ImageId'
#df.index+=1
#df.columns=['Label']
#df.to_csv('result-svm.csv', header=True)



#Principal component analysis
print("Principal component analysis")
pca = PCA(n_components=0.8, whiten=True)
pca.fit(X)
X = pca.transform(X)
X_test = pca.transform(X_test)

print("\nSupport vector classifier 0.8 variance of reduced dimensions")
#grid search for best set of hyper parameters
parameters = {'kernel':['rbf'], 'C':np.logspace(-5,15,21,base=2),
'gamma':np.logspace(-15,3,19,base=2), 'epsilon':np.linspace(0,0.5,11)}
svr = SVR()
clf = GridSearchCV(svr, parameters, n_jobs=-1, verbose=2, cv=3)
clf.fit(X, y)
best = clf.best_params_
print("Best parameters are:")
print(best)
#past best were: 
#best = {'epsilon': 0.0, 'C': 32.0, 'gamma': 0.001953125}

tr_perf=[]
val_perf=[]
#10-fold cross validation on data
kf = KFold(n_splits=10)
fold=0;
for train_index, test_index in kf.split(X):
  X_train, X_val = X[train_index], X[test_index]
  y_train, y_val = y[train_index], y[test_index]

  #Random Forest algorithm
  svr = SVR(kernel='rbf', epsilon=best['epsilon'], C=best['C'], gamma=best['gamma'])
  svr.fit(X_train, y_train)
  pred = np.around(svr.predict(X_train)).astype(int)
  pred[pred<0]=0
  pred[pred>9]=9
  tr_perf.append(accuracy_score(pred, y_train))
  print("Accuracy on training set fold: {:.3f}".format(tr_perf[fold]))
  pred = np.around(svr.predict(X_val)).astype(int)
  pred[pred<0]=0
  pred[pred>9]=9
  val_perf.append(accuracy_score(pred,y_val))
  print("Accuracy on validation set fold: {:.3f}".format(val_perf[fold]))
  fold += 1
print("Average accuracy on training set fold: {:.3f} +/-{:.3f}".format(np.mean(tr_perf),np.std(tr_perf)))
print("Average accuracy on validation set fold: {:.3f} +/-{:.3f}".format(np.mean(val_perf),np.std(val_perf)))

#Random Forest algorithm
svr = SVR(kernel='rbf', epsilon=best['epsilon'], C=best['C'], max_iter=-1, verbose=True)
svr.fit(X_train, y_train)
pred = np.around(svr.predict(X_train)).astype(int)
pred[pred<0]=0
pred[pred>9]=9
print("Accuracy on full training set: {:.3f}".format(accuracy_score(pred, y_train)))
result = np.around(svr.predict(X_test)).astype(int)
result[result<0]=0
result[result>9]=9
#To save results into .csv file and submit to kaggle
df = pd.DataFrame(result)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('result-svm-pca.csv', header=True)



