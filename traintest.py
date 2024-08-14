import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Define the token and API URL
TOKEN = "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJzdXBlcmFkbWluQGdtYWlsLmNvbSIsImlhdCI6MTcyMzYzMjYzNCwiZXhwIjoxNzI0MjM3NDM0fQ.6yzghOZl1A8YMhOg7eUDi68egc9oRrr0s-CAy9oLvu-038YFbdo-Ee9IovyQubsvIlpsvtSPbjcGYmA3LQkLiA"
API_URL = "https://authentication.tawasalna.com/tawasalna-user/user/"

def load_data():
    headers = {'Authorization': f'Bearer {TOKEN}'}
    try:
        response = requests.get(API_URL, headers=headers)
        response.raise_for_status()
        data = pd.json_normalize(response.json())
        
        # Print columns to understand the structure
        print("Columns in the fetched data:", data.columns.tolist())
        
        return data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"Other error occurred: {err}")
        return None

def prepare_features(df):
    # Print the first few rows to understand the data
    print("First few rows of the data:")
    print(df.head())
    
    # Selecting relevant columns (adjust based on actual column names)
    if 'address' not in df.columns or 'username' not in df.columns:
        raise KeyError("Required columns are not present in the data.")
    
    # Use available columns, e.g., 'address' and 'username' for features
    df = df[['address', 'username']]
    
    # Handle missing values
    df = df.fillna('missing')
    
    # Convert categorical data to numerical data
    label_encoders = {}
    for column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # Prepare features
    X = df[['address']].values  # Assuming 'address' as a feature
    return X, df

def cluster_and_find_neighbors(X, df):
    # Initialize KMeans model for clustering
    n_clusters = 5  # Define number of clusters, adjust as needed
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    
    # Assign cluster labels to the data
    df['cluster'] = kmeans.labels_
    
    # Initialize KNN model
    knn = NearestNeighbors(n_neighbors=5)  # You can change the number of neighbors
    knn.fit(X)
    
    return knn, df

def find_closest_users(knn, X, user_index, n_neighbors=5):
    """
    Find the closest users to a given user.
    
    Parameters:
    - user_index: Index of the user for whom we want to find the nearest neighbors
    - n_neighbors: Number of neighbors to find
    
    Returns:
    - indices of the closest users
    """
    distances, indices = knn.kneighbors([X[user_index]], n_neighbors=n_neighbors)
    return indices[0]

def calculate_model_efficacy(X, clusters):
    """
    Calculate clustering model efficacy using Silhouette Score.
    
    Returns:
    - silhouette_score: Measure of how similar an object is to its own cluster compared to other clusters
    """
    # Calculate Silhouette Score
    score = silhouette_score(X, clusters)
    return score

def generate_html(closest_users_indices, data, model_efficacy):
    """
    Generate an HTML report for the closest users and model efficacy.
    
    Parameters:
    - closest_users_indices: List of indices for the closest users
    - data: DataFrame containing user data
    - model_efficacy: Silhouette score of the clustering model
    """
    html_content = '''
    <html>
    <head>
        <title>Closest Users</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { padding: 10px; text-align: left; border: 1px solid #ddd; }
            th { background-color: #f4f4f4; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .efficacy { margin-top: 20px; padding: 10px; border: 1px solid #ddd; background-color: #f9f9f9; }
        </style>
    </head>
    <body>
        <h1>Closest Users to User at Index 0</h1>
        <table>
            <tr>
                <th>Username</th>
            </tr>'''
    
    for idx in closest_users_indices:
        username = data.iloc[idx]['username']
        html_content += f'<tr><td>{username}</td></tr>'
    
    html_content += '''
        </table>
        <div class="efficacy">
            <h2>Model Efficacy</h2>
            <p><strong>Silhouette Score:</strong> {:.2f}</p>
        </div>
    </body>
    </html>'''.format(model_efficacy)

    # Save HTML to file
    with open('closest_users.html', 'w') as file:
        file.write(html_content)
    print("HTML file 'closest_users.html' has been created.")

# Main execution
data = load_data()
if data is not None:
    try:
        X, df = prepare_features(data)
        knn, df = cluster_and_find_neighbors(X, df)
        
        # Example: Find closest users to the user at index 0
        closest_users_indices = find_closest_users(knn, X, user_index=0, n_neighbors=5)
        
        # Calculate model efficacy
        model_efficacy = calculate_model_efficacy(X, df['cluster'])
        
        # Generate HTML content
        generate_html(closest_users_indices, data, model_efficacy)
    except KeyError as e:
        print(f"Error: {e}")
else:
    print("Failed to load data.")
