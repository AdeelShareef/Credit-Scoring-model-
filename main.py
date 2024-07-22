
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('credit_data.csv')

# Explore the dataset
print(data.head())
print(data.info())
print(data.describe())

# Handle missing values (e.g., fill with mean or median)
data.fillna(data.mean(), inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split features and target variable
X = data.drop('creditworthy', axis=1)  # Assuming 'creditworthy' is the target variable
y = data['creditworthy']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Train a Logistic Regression model for comparison
log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
rf_pred = rf_model.predict(X_test_scaled)
log_reg_pred = log_reg_model.predict(X_test_scaled)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_pred)
rf_class_report = classification_report(y_test, rf_pred)

print('Random Forest Model:')
print(f'Accuracy: {rf_accuracy}')
print('Confusion Matrix:')
print(rf_conf_matrix)
print('Classification Report:')
print(rf_class_report)

# Evaluate the Logistic Regression model
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
log_reg_conf_matrix = confusion_matrix(y_test, log_reg_pred)
log_reg_class_report = classification_report(y_test, log_reg_pred)

print('Logistic Regression Model:')
print(f'Accuracy: {log_reg_accuracy}')
print('Confusion Matrix:')
print(log_reg_conf_matrix)
print('Classification Report:')
print(log_reg_class_report)

# Plot the confusion matrix for Random Forest
plt.figure(figsize=(8, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Plot the confusion matrix for Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(log_reg_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# Feature Importance for Random Forest
rf_feature_importances = rf_model.feature_importances_
rf_feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importances for Random Forest
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=rf_feature_importance_df)
plt.title('Random Forest Feature Importances')
plt.show()
