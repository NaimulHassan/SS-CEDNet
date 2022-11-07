# coding= UTF-8
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier #Random Forest classifier
import pandas as pd
import numpy as np
np.random.seed(0)

#Load data
X_train = np.load('train_feat.npy')
y_train = np.load('train_label.npy').ravel()

X_test = np.load('eval_feat.npy')
y_test = np.load('eval_label.npy').ravel()

# Initialize classifier
rf_clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train model
rf_clf.fit(X_train, y_train)

# Make predictions
y_prediction = rf_clf.predict(X_test)

#print('Predicted values')
#print(y_prediction)
#print
#print('Actual values')
#print(y_test)
#print
#print(y_prediction-y_test)

# Evaluate accuracy
print
acc = rf_clf.score(X_test, y_test)
print("Accuracy = %0.3f" %acc)

# View the predicted probabilities of the first n observations
rf_clf.predict_proba(X_test)[0:10]

# For  label decoding
label_classes = np.array(['Cough','TIMIT'])
#print(label_classes)

# Decoding predicted and actual classes (numeric to written)
prediction_decoded = label_classes[y_prediction]
actual_value_decoded = label_classes[y_test]

## Generate Confusion Matrix
print(pd.crosstab(actual_value_decoded, prediction_decoded))

print(precision_recall_fscore_support(y_test, y_prediction, average=None))