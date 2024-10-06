from flask import Flask, render_template_string
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

app = Flask(__name__)

# Load model and scaler
pipeline = joblib.load('fraud_detection_pipeline.pkl')
scaler = joblib.load('scaler.pkl')

# Define the Flask route
@app.route('/')
def fraud_detection():
    # Prepare new data for prediction
    new_data = pd.DataFrame({
        'amount': [500.00, 1000000.00, 200.00, 5000.00, 750.00, 1200.00, 300.00, 45000.00, 123.00, 20.00],
        'transaction_type': ['online', 'online', 'online', 'offline', 'online', 'offline', 'online', 'online', 'offline', 'offline'],
        'merchant_id': [1234, 5678, 67890, 13579, 24680, 112233, 445566, 778899, 101112, 131415],
        'account_age_days': [60, 10, 300, 5, 2, 100, 150, 20, 500, 1],
        'transaction_hour': [23, 14, 8, 18, 3, 21, 12, 6, 20, 15]
    })

    # Encode categorical variables and align columns
    X_train = pd.read_csv('transactions.csv')  # Use training data columns for alignment
    X_train = pd.get_dummies(X_train.drop('is_fraud', axis=1), drop_first=True)
    new_data = pd.get_dummies(new_data, drop_first=True)
    new_data = new_data.reindex(columns=X_train.columns, fill_value=0)
    new_data_scaled = scaler.transform(new_data)

    # Make predictions
    predictions = pipeline.predict(new_data_scaled)

    # Prepare results
    results_df = pd.DataFrame({
        'amount': [500.00, 1000000.00, 200.00, 5000.00, 750.00, 1200.00, 300.00, 45000.00, 123.00, 20.00],
        'transaction_type': ['online', 'online', 'online', 'offline', 'online', 'offline', 'online', 'online', 'offline', 'offline'],
        'merchant_id': [1234, 5678, 67890, 13579, 24680, 112233, 445566, 778899, 101112, 131415],
        'account_age_days': [60, 10, 300, 5, 2, 100, 150, 20, 500, 1],
        'transaction_hour': [23, 14, 8, 18, 3, 21, 12, 6, 20, 15],
        'prediction': ['Fraudulent' if p == 1 else 'Not Fraudulent' for p in predictions]
    })

    # Generate HTML table for results
    table_html = results_df.to_html(index=False)

    # Evaluation (example using the test set split during training)
    y_test = pd.read_csv('y_test.csv')  # Assuming you saved y_test from earlier
    y_pred = pipeline.predict(scaler.transform(pd.get_dummies(X_train, drop_first=True)))
    class_report = classification_report(y_test, y_pred)

    # Combine all into an HTML page
    html_content = f'''
    <html>
    <head>
        <title>Fraud Detection Results</title>
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            table, th, td {{
                border: 1px solid black;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h1>Fraud Detection Prediction Results</h1>
        {table_html}
        <h2>Classification Report</h2>
        <pre>{class_report}</pre>
    </body>
    </html>
    '''
    
    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5007)
