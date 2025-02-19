import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb  # Make sure you have xgboost installed: pip install xgboost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and Prepare Data
data = pd.read_csv(r"C:\Users\91944\Desktop\Data_Analyst_Project\Bank Customer Churn Prediction.csv")

# Convert categorical features to numerical
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data = pd.get_dummies(data, columns=['country'], drop_first=True)  # Creates France=0, Germany=0/1, Spain=0

# Define features (X) and target (y)
X = data.drop(['customer_id', 'churn'], axis=1)
y = data['churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42) # Adjust hyperparameters as needed
model.fit(X_train, y_train)

# 3. Model Prediction and Evaluation
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Random Forest Model Performance:")
print(f"Accuracy: {accuracy:.4f}")  # Format to 4 decimal places
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)

# (Optional) Visualize the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
