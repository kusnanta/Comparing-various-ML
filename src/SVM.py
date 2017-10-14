import horsesIntro


X_train = horsesIntro.X_train
y_train = horsesIntro.y_train
X_test = horsesIntro.X_test
y_test = horsesIntro.y_test

from sklearn import metrics
from sklearn.svm import SVC
svm_model= SVC()

tuned_parameters = {
 'C': [1, 10, 100,500, 1000], 'kernel': ['linear','rbf'],
 'C': [1, 10, 100,500, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf'],
 #'degree': [2,3,4,5,6] , 'C':[1,10,100,500,1000] , 'kernel':['poly']
    }

from sklearn.grid_search import RandomizedSearchCV

model_svm = RandomizedSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy',n_iter=20)

model_svm.fit(X_train, y_train)
print(model_svm.best_score_)

print(model_svm.grid_scores_)

print(model_svm.best_params_)

y_pred= model_svm.predict(X_test)
print(metrics.accuracy_score(y_pred,y_test))

confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
print(confusion_matrix)

auc_roc=metrics.classification_report(y_test,y_pred)
print(auc_roc)

auc_roc=metrics.roc_auc_score(y_test,y_pred)
print(auc_roc)

from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()