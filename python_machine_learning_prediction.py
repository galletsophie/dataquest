## 1. Recap ##

import pandas as pd

loans = pd.read_csv('cleaned_loans_2007.csv')
print(loans.info())

## 3. Picking an error metric ##

import pandas as pd

tn_filter = (predictions == 0) & (loans['loan_status']==0)
tp_filter = (predictions == 1) & (loans['loan_status']==1)
fn_filter = (predictions == 0) & (loans['loan_status']==1)
fp_filter = (predictions == 1) & (loans['loan_status']==0)

tn = len(predictions[tn_filter])
tp = len(predictions[tp_filter])
fn = len(predictions[fn_filter])
fp = len(predictions[fp_filter])


## 5. Class imbalance ##

import pandas as pd
import numpy

# Predict that all loans will be paid off on time.
predictions = pd.Series(numpy.ones(loans.shape[0]))

tn_filter = (predictions == 0) & (loans["loan_status"]==0)
tp_filter = (predictions == 1) & (loans['loan_status']==1)
fn_filter = (predictions == 0) & (loans['loan_status']==1)
fp_filter = (predictions == 1) & (loans['loan_status']==0)

tn = len(predictions[tn_filter])
tp = len(predictions[tp_filter])
fn = len(predictions[fn_filter])
fp = len(predictions[fp_filter])

fpr = fp / (fp+tn)
tpr = tp / (tp+fn)

## 6. Logistic Regression ##

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

features = loans.drop(['loan_status'], axis=1)
target = loans['loan_status']
lr.fit(features, target)
predictions = lr.predict(features)

## 7. Cross Validation ##

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
lr = LogisticRegression()

predictions = pd.Series(cross_val_predict(lr, features, target, cv = 3))

tp_filter = (predictions == 1) & (loans['loan_status']==1)
tp = len(predictions[tp_filter])

tn_filter = (predictions == 0) & (loans['loan_status']==0)
tn = len(predictions[tn_filter])

fp_filter = (predictions == 1) & (loans['loan_status']==0)
fp = len(predictions[fp_filter])

fn_filter = (predictions == 0) & (loans['loan_status']==1)
fn = len(predictions[fn_filter])

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)



## 9. Penalizing the classifier ##

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

lr = LogisticRegression(class_weight = 'balanced')
predictions = pd.Series(cross_val_predict(lr, features, target, cv=3))

tp_filter = (predictions == 1) & (loans['loan_status']==1)
tp = len(predictions[tp_filter])

tn_filter = (predictions == 0) & (loans['loan_status']==0)
tn = len(predictions[tn_filter])

fp_filter = (predictions == 1) & (loans['loan_status']==0)
fp = len(predictions[fp_filter])

fn_filter = (predictions == 0) & (loans['loan_status']==1)
fn = len(predictions[fn_filter])

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)


## 10. Manual penalties ##

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

penalty = {
    0:10,
    1:1
}


lr = LogisticRegression(class_weight = penalty)
predictions = pd.Series(cross_val_predict(lr, features, target, cv=3))

tp_filter = (predictions == 1) & (loans['loan_status']==1)
tp = len(predictions[tp_filter])

tn_filter = (predictions == 0) & (loans['loan_status']==0)
tn = len(predictions[tn_filter])

fp_filter = (predictions == 1) & (loans['loan_status']==0)
fp = len(predictions[fp_filter])

fn_filter = (predictions == 0) & (loans['loan_status']==1)
fn = len(predictions[fn_filter])

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)



## 11. Random forests ##

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict

rf = RandomForestClassifier(class_weight = 'balanced', random_state =1)
predictions = pd.Series(cross_val_predict(rf, features, target, cv=3))

tp_filter = (predictions == 1) & (loans['loan_status']==1)
tp = len(predictions[tp_filter])

tn_filter = (predictions == 0) & (loans['loan_status']==0)
tn = len(predictions[tn_filter])

fp_filter = (predictions == 1) & (loans['loan_status']==0)
fp = len(predictions[fp_filter])

fn_filter = (predictions == 0) & (loans['loan_status']==1)
fn = len(predictions[fn_filter])

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)


penalty = {
    0:10,
    1:1
}

rf2 = RandomForestClassifier(class_weight = penalty, random_state =1)
predictions = pd.Series(cross_val_predict(rf2, features, target, cv=3))

tp_filter = (predictions == 1) & (loans['loan_status']==1)
tp2 = len(predictions[tp_filter])

tn_filter = (predictions == 0) & (loans['loan_status']==0)
tn2 = len(predictions[tn_filter])

fp_filter = (predictions == 1) & (loans['loan_status']==0)
fp2 = len(predictions[fp_filter])

fn_filter = (predictions == 0) & (loans['loan_status']==1)
fn2 = len(predictions[fn_filter])

tpr2 = tp2 / (tp2 + fn2)
fpr2 = fp2 / (fp2 + tn2)

print(tpr2, fpr2)

