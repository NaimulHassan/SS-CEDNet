from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier #Random Forest classifier

#Load data
X_train = np.load('train_feat.npy')
y_train = np.load('train_label.npy').ravel()

X_test = np.load('mixed_feat.npy')
y_test = np.load('mixed_label.npy').ravel()

# Initialize classifier
gnb_clf= GaussianNB() #check input params

# Train model
gnb_clf.fit(X_train, y_train)

# Make predictions
prediction = gnb_clf.predict(X_test)

#print('Predicted values')
#print(prediction)
#print
#print('Actual values')
#print(y_test)

# Evaluate accuracy
print
acc = gnb_clf.score(X_test, y_test)
print("Accuracy = %0.3f" %acc)
#print(accuracy_score(y_test,prediction)) # Equivalent way to do it

label_classes = np.array(['Cough','TIMIT'])
#print(label_classes)

# Decoding predicted and actual classes (numeric to written)
prediction_decoded = label_classes[prediction]
actual_value_decoded = label_classes[y_test]

## Generate Confusion Matrix
print(pd.crosstab(actual_value_decoded, prediction_decoded))

print(precision_recall_fscore_support(y_test, prediction, average=None))