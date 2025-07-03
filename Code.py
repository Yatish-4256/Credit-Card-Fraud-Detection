
# Install Required Libraries (Run in terminal if not installed)
# pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # For handling class imbalance

#  Load Dataset 
# Replace with your file path
data = pd.read_csv('creditcard.csv')


print("\n=== Dataset Head ===")
print(data.head())

print("\n=== Class Distribution ===")
print(data['Class'].value_counts())


plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=data)
plt.title('Fraud (1) vs Non-Fraud (0) Transactions')
plt.show()


# Check for missing values
print("\n=== Missing Values ===")
print(data.isnull().sum())

# Normalize 'Amount' column
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

# Drop 'Time' column
data = data.drop(['Time'], axis=1)

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# 6. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Handle Class Imbalance with SMOTE (Optional)
print("\nApplying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# 8. Train Model
print("\nTraining Logistic Regression Model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_res, y_res)  # Use X_train, y_train if not using SMOTE

# 9. Evaluate Model
y_pred = model.predict(X_test)

print("\n=== Model Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,4))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# 10. Predict on New Data (Example)
# new_transaction = [[-1.359807, -0.072781, 2.536347, ..., 0.133558, -0.021053, 0.5]]
# prediction = model.predict(new_transaction)
# print("Fraud Prediction (1=Fraud, 0=Legitimate):", prediction[0])
