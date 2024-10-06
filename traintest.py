from flask import Flask, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved model and scaler
pipeline = joblib.load('fraud_detection_pipeline.pkl')
scaler = joblib.load('scaler.pkl')

# Load the transaction data
data = pd.read_csv('transactions.csv')

# Preprocess data
def preprocess_data(data):
    # Convert transaction_time to relevant features
    data['transaction_hour'] = pd.to_datetime(data['transaction_time'], format='%H:%M:%S').dt.hour
    data.drop('transaction_time', axis=1, inplace=True)

    target_column = 'is_fraud'  # Ensure this matches your dataset
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    return X, y

X, y = preprocess_data(data)

# Dummy data for prediction
def get_new_data(X):
    new_data = pd.DataFrame({
        'amount': [500.00, 1000000.00, 200.00, 5000.00, 750.00, 1200.00, 300.00, 45000.00, 123.00, 20.00],
        'transaction_type': ['online', 'online', 'online', 'offline', 'online', 'offline', 'online', 'online', 'offline', 'offline'],
        'merchant_id': [1234, 5678, 67890, 13579, 24680, 112233, 445566, 778899, 101112, 131415],
        'account_age_days': [60, 10, 300, 5, 2, 100, 150, 20, 500, 1],
        'transaction_hour': [23, 14, 8, 18, 3, 21, 12, 6, 20, 15]
    })

    # Encode categorical variables and align columns
    X_new = pd.get_dummies(new_data, drop_first=True)
    X_new = X_new.reindex(columns=X.columns, fill_value=0)  # Align new data with training data
    X_scaled = scaler.transform(X_new)

    return X_scaled, new_data

@app.route('/')
def predict_fraud():
    # Get new data and predictions
    X_new, original_data = get_new_data(X)
    predictions = pipeline.predict(X_new)

    # Prepare results
    original_data['prediction'] = ['Fraudulent' if p == 1 else 'Not Fraudulent' for p in predictions]

    # Create HTML table
    results_html = original_data.to_html(classes='table table-striped', index=False)

    # Render HTML template
    html_template = '''
    <html>
    <head>
        <title>Fraud Detection Predictions</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
    </head>
    <body>
        <div class="container">
            <h1 class="mt-5">Fraud Detection Prediction Results</h1>
            {{ table | safe }}
        </div>
    </body>
    </html>
    '''
    return render_template_string(html_template, table=results_html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007)
