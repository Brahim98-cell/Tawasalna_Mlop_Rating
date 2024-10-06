from flask import Flask, render_template_string, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load pre-trained model and scaler
pipeline = joblib.load('fraud_detection_pipeline.pkl')
scaler = joblib.load('scaler.pkl')

# Load the data from transactions.csv
def load_transaction_data():
    data = pd.read_csv('transactions.csv')

    # Convert transaction_time to relevant features
    data['transaction_hour'] = pd.to_datetime(data['transaction_time'], format='%H:%M:%S').dt.hour
    data.drop('transaction_time', axis=1, inplace=True)

    # Encode categorical variables
    data = pd.get_dummies(data, drop_first=True)

    return data

@app.route('/')
def fraud_detection():
    # Load transaction data
    new_data = load_transaction_data()

    # Align columns with the trained model input
    new_data = new_data.reindex(columns=X.columns, fill_value=0)

    # Scale the input data
    new_data_scaled = scaler.transform(new_data)

    # Make predictions
    predictions = pipeline.predict(new_data_scaled)

    # Append predictions to the DataFrame
    new_data['prediction'] = ['Fraudulent' if pred == 1 else 'Not Fraudulent' for pred in predictions]

    # Generate HTML content dynamically
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
    for index, row in new_data.iterrows():
        html_content += f'''
            <tr>
                <td>{row.get('amount', 'N/A')}</td>
                <td>{row.get('transaction_type', 'N/A')}</td>
                <td>{row.get('merchant_id', 'N/A')}</td>
                <td>{row.get('account_age_days', 'N/A')}</td>
                <td>{row.get('transaction_hour', 'N/A')}</td>
                <td>{row['prediction']}</td>
            </tr>
        '''
    
    html_content += '''
        </table>
    </body>
    </html>
    '''

    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5007)
