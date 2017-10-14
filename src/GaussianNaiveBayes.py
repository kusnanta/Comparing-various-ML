import horsesIntro


X_train = horsesIntro.X_train
y_train = horsesIntro.y_train
X_test = horsesIntro.X_test
y_test = horsesIntro.y_test
np = horsesIntro.np
X = horsesIntro.X
y = horsesIntro.y

from sklearn.model_selection import cross_val_score
from sklearn import metrics

from sklearn.naive_bayes import GaussianNB
model_naive = GaussianNB()
print(model_naive.fit(X_train, y_train))

y_prob = model_naive.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
print(model_naive.score(X_test, y_pred))

print("Number of mislabeled points from %d points : %d"
      % (X_test.shape[0],(y_test!= y_pred).sum()))

scores = cross_val_score(model_naive, X, y, cv=10, scoring='accuracy')
print(scores)

print(scores.mean())

confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
print(confusion_matrix)

auc_roc=metrics.classification_report(y_test,y_pred)
print(auc_roc)

auc_roc=metrics.roc_auc_score(y_test,y_pred)
print(auc_roc)

from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
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