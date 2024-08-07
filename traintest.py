import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
from io import StringIO

# Load and preprocess data
data = pd.read_csv('transactions.csv')

# Convert transaction_time to relevant features
data['transaction_hour'] = pd.to_datetime(data['transaction_time'], format='%H:%M:%S').dt.hour
data.drop('transaction_time', axis=1, inplace=True)

target_column = 'is_fraud'  # Ensure this matches your dataset
X = data.drop(target_column, axis=1)
y = data[target_column]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize SMOTE and Random Forest
smote = SMOTE(random_state=42, k_neighbors=min(2, len(X_train) - 1))
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Create and train the pipeline
pipeline = Pipeline([
    ('smote', smote),
    ('classifier', classifier)
])
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(pipeline, 'fraud_detection_pipeline.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load model and scaler
loaded_pipeline = joblib.load('fraud_detection_pipeline.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Prepare new data for prediction
new_data = pd.DataFrame({
    'amount': [500.00, 1000000.00, 200.00, 5000.00, 750.00, 1200.00, 300.00, 45000.00, 123.00, 20.00],
    'transaction_type': ['online', 'online', 'online', 'offline', 'online', 'offline', 'online', 'online', 'offline', 'offline'],
    'merchant_id': [1234, 5678, 67890, 13579, 24680, 112233, 445566, 778899, 101112, 131415],
    'account_age_days': [60, 10, 300, 5, 2, 100, 150, 20, 500, 1],
    'transaction_hour': [23, 14, 8, 18, 3, 21, 12, 6, 20, 15]
})

# Encode categorical variables and align columns
new_data = pd.get_dummies(new_data, drop_first=True)
new_data = new_data.reindex(columns=X.columns, fill_value=0)
new_data = loaded_scaler.transform(new_data)

# Make predictions
predictions = loaded_pipeline.predict(new_data)

# Prepare results for HTML
results_df = pd.DataFrame({
    'amount': [500.00, 1000000.00, 200.00, 5000.00, 750.00, 1200.00, 300.00, 45000.00, 123.00, 20.00],
    'transaction_type': ['online', 'online', 'online', 'offline', 'online', 'offline', 'online', 'online', 'offline', 'offline'],
    'merchant_id': [1234, 5678, 67890, 13579, 24680, 112233, 445566, 778899, 101112, 131415],
    'account_age_days': [60, 10, 300, 5, 2, 100, 150, 20, 500, 1],
    'transaction_hour': [23, 14, 8, 18, 3, 21, 12, 6, 20, 15],
    'prediction': ['Fraudulent' if p == 1 else 'Not Fraudulent' for p in predictions]
})

# Convert classification report to HTML
report_html = '<h2>Classification Report</h2>'
report_html += '<pre>' + classification_report(y_test, y_pred) + '</pre>'

# Generate HTML content
html_content = '''
<html>
<head>
    <title>Prediction Results</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table, th, td {
            border: 1px solid black;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <h1>Fraud Detection Prediction Results</h1>
    <table>
        <tr>
            <th>Amount</th>
            <th>Transaction Type</th>
            <th>Merchant ID</th>
            <th>Account Age (days)</th>
            <th>Transaction Hour</th>
            <th>Prediction</th>
        </tr>
'''

# Append rows for each prediction
for index, row in results_df.iterrows():
    html_content += f'''
        <tr>
            <td>{row['amount']}</td>
            <td>{row['transaction_type']}</td>
            <td>{row['merchant_id']}</td>
            <td>{row['account_age_days']}</td>
            <td>{row['transaction_hour']}</td>
            <td>{row['prediction']}</td>
        </tr>
    '''

html_content += '''
    </table>
    ''' + report_html + '''
</body>
</html>
'''

# Write HTML content to file
with open('prediction_results.html', 'w') as f:
    f.write(html_content)

print("\nPredictions and classification report saved to prediction_results.html")
