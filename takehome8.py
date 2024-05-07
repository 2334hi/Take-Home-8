import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score

glass = pd.read_csv("glass.csv")
#print(glass.head())

glass_sorts = glass.Type.value_counts().sort_index()
#print(glass_sorts) # Window Glass Types: 1, 2, 3    => 163 Window Glass
                   # Household Glass Types: 5, 6, 7  => 51 Household Glass
glass['household'] = glass.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1}) # Normalization of Data, household is our hot encoding
glass.household.value_counts()

glass.sort_values( by= "Al", inplace = True)
X= np.array(glass.Al).reshape(-1,1)
y = glass.household

logreg = LogisticRegression()
logreg.fit(X,y)
pred = logreg.predict(X)
logreg.coef_, logreg.intercept_
glass.sort_values( by = 'Al', inplace=True)



# Prediction with default thershold
cm = metrics.confusion_matrix(y_true=y, y_pred=pred)
Accuracy = (cm[0,0]+ cm[1,1])/ (np.sum(cm))
Precision = (cm[1,1])/ (np.sum(cm[: , 1]))
Recall = (cm[1,1])/ (np.sum(cm[1,:]))
print("Accuracy, Precision, and Recall at thershold = 0.5")
print(Accuracy)
print(Precision)
print(Recall)
#print(first_30_preds)

# Prediction with thershold of 0.9
X = X[0:30]
y = y[0:30]
first_30_preds = (logreg.predict_proba(X) >= 0.9)

print(len(X))
print(len(y))
# PROBLEMS with figuring out the label sizes
#accuracy_score(y_true=y, y_pred=first_30_preds) # Idea is to try out a new prediction thershold
# cm = metrics.confusion_matrix(y_true=y, y_pred=pred)
# Accuracy = (cm[0,0]+ cm[1,1])/ (np.sum(cm))
# Precision = (cm[1,1])/ (np.sum(cm[: , 1]))
# Recall = (cm[1,1])/ (np.sum(cm[1,:]))
# print("Accuracy, Precision, and Recall at thershold = 0.5")
# print(Accuracy)
# print(Precision)
# print(Recall)
#print(first_30_preds)

# Analysis for Ca at 0.5
glass.sort_values( by = "Ca", inplace = True)
X = np.array(glass.Ca).reshape(-1, 1)
y = glass.household
logreg = LogisticRegression()
logreg.fit(X,y)
pred = logreg.predict(X)
logreg.coef_, logreg.intercept_
glass.sort_values( by = 'Ca', inplace=True)

cm2 = metrics.confusion_matrix(y_true=y, y_pred=pred)
Accuracy = (cm2[0,0]+ cm2[1,1])/ (np.sum(cm2))
Precision = (cm2[1,1])/ (np.sum(cm2[: , 1]))
Recall = (cm2[1,1])/ (np.sum(cm2[1,:]))
print("Accuracy, Precision, and Recall at thershold = 0.5")
print(Accuracy)
print(Precision)
print(Recall)

# Analysis for Si at 0.5
glass.sort_values( by = "Si", inplace = True)
X = np.array(glass.Si).reshape(-1, 1)
y = glass.household
logreg = LogisticRegression()
logreg.fit(X,y)
pred = logreg.predict(X)
logreg.coef_, logreg.intercept_
glass.sort_values( by = 'Si', inplace=True)

cm = metrics.confusion_matrix(y_true=y, y_pred=pred)
Accuracy = (cm[0,0]+ cm[1,1])/ (np.sum(cm))
Precision = (cm[1,1])/ (np.sum(cm[: , 1]))
Recall = (cm[1,1])/ (np.sum(cm[1,:]))
print("Accuracy, Precision, and Recall at thershold = 0.5")
print(Accuracy)
print(Precision)
print(Recall)




