# coding= UTF-8
import numpy as np
import sklearn
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier #Random Forest classifier
import pandas as pd

#Load data
X_train = np.load('train_feat.npy')
y_train = np.load('train_label.npy').ravel()

X_test = np.load('eval_feat.npy')
y_test = np.load('eval_label.npy').ravel()
# Data scaling (NOT IMPLEMENTING)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))

# Implement simple linear SVM
svm_clf = SVC(C=28.0, gamma = 0.00001, decision_function_shape="ovr") #These parameters can be modified

# Fit model
svm_clf.fit(X_train, y_train) #From Beif github
#svm_clf.fit(X_train_scaled, y_train) # HandsOn book

# Make predictions
#y_pred = svm_clf.predict(X_train_scaled)
y_predict = svm_clf.predict(X_test)

#print('Prediction')
#print(y_predict)
#print
#print("Actual")
#print(y_test)

# Accuracy
acc = svm_clf.score(X_test, y_test)
print
print("accuracy=%0.3f" %acc)


label_classes = np.array(['Cough','TIMIT'])
#print(label_classes)

# Decoding predicted and actual classes (numeric to written)
prediction_decoded = label_classes[y_predict]
actual_value_decoded = label_classes[y_test]

## Generate Confusion Matrix
print(pd.crosstab(actual_value_decoded, prediction_decoded))

print(precision_recall_fscore_support(y_test, y_predict, average=None))