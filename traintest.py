import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib

# Load data
data = pd.read_csv('transactions.csv')

# Print column names to verify
print("Columns in dataset:", data.columns)

# Convert transaction_time to relevant features
data['transaction_hour'] = pd.to_datetime(data['transaction_time'], format='%H:%M:%S').dt.hour
data.drop('transaction_time', axis=1, inplace=True)

# Ensure the correct target column name
target_column = 'is_fraud'  # Update this with the actual target column name

# Separate features and target
X = data.drop(target_column, axis=1)  # Features
y = data[target_column]  # Target

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)  # Convert categorical features to dummy variables

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize SMOTE and Random Forest classifier
# Ensure k_neighbors <= number of samples in training set
smote = SMOTE(random_state=42, k_neighbors=min(2, len(X_train) - 1))  # Adjust k_neighbors
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a pipeline with SMOTE and Random Forest
pipeline = Pipeline([
    ('smote', smote),
    ('classifier', classifier)
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and scaler to files
joblib.dump(pipeline, 'fraud_detection_pipeline.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load the model and scaler
loaded_pipeline = joblib.load('fraud_detection_pipeline.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Prepare new data for prediction
new_data = pd.DataFrame({
    'amount': [500.00],  # Extremely high value
    'transaction_type': ['online'],  # Example categorical feature
    'merchant_id': [1234],  # Use a different ID
    'account_age_days': [60],  # Very new account
    'transaction_hour': [23]  # Example feature (converted transaction time)
})

# Encode categorical variables in new data
new_data = pd.get_dummies(new_data, drop_first=True)

# Align new data with model features
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# Scale new data
new_data = loaded_scaler.transform(new_data)

# Make predictions
predictions = loaded_pipeline.predict(new_data)
print("\nPredictions for new data:")
print(predictions)
