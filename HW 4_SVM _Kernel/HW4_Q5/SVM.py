import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('sonar_csv.csv')

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42,shuffle=True)

# Separate features and labels
X_train, y_train = train_data.drop('Class', axis=1), train_data['Class']
X_test, y_test = test_data.drop('Class', axis=1), test_data['Class']
# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

# Polynomial Kernel
svm_poly = SVC(kernel='poly', degree=3)  
svm_poly.fit(X_train, y_train)

# RBF Kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)

# predication train
y_pred_linear_train = svm_linear.predict(X_train)
y_pred_poly_train = svm_poly.predict(X_train)
y_pred_rbf_train = svm_rbf.predict(X_train)

# predication test
y_pred_linear = svm_linear.predict(X_test)
y_pred_poly = svm_poly.predict(X_test)
y_pred_rbf = svm_rbf.predict(X_test)

# Accuracy train
acc_linear_train = accuracy_score(y_train, y_pred_linear_train)
acc_poly_train = accuracy_score(y_train, y_pred_poly_train)
acc_rbf_train = accuracy_score(y_train, y_pred_rbf_train)

# Accuracy test
acc_linear_test = accuracy_score(y_test, y_pred_linear)
acc_poly_test = accuracy_score(y_test, y_pred_poly)
acc_rbf_test = accuracy_score(y_test, y_pred_rbf)
print(f'___________________________________________\nAccuracy in Train')
print(f'Linear Kernel Accuracy: {acc_linear_train}')
print(f'Polynomial Kernel Accuracy: {acc_poly_train}')
print(f'RBF Kernel Accuracy: {acc_rbf_train}')
print(f'___________________________________________\nAccuracy in Test')
print(f'Linear Kernel Accuracy: {acc_linear_test}')
print(f'Polynomial Kernel Accuracy: {acc_poly_test}')
print(f'RBF Kernel Accuracy: {acc_rbf_test}')