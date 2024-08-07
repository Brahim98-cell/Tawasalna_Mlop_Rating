import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Load your CSV data
data = pd.read_csv('TawasalnaDB.users.csv')

# Selecting relevant columns
df = data[['address', 'community', 'username']]

# Handle missing values
df = df.fillna('missing')

# Convert categorical data to numerical data
label_encoders = {}
for column in df.columns[:-1]:  # Exclude 'username' column
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Prepare features
X = df[['address', 'community']].values

# Initialize KMeans model for clustering
n_clusters = 5  # Define number of clusters, adjust as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Assign cluster labels to the data
df['cluster'] = kmeans.labels_

# Initialize KNN model
knn = NearestNeighbors(n_neighbors=5)  # You can change the number of neighbors
knn.fit(X)

def find_closest_users(user_index, n_neighbors=5):
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

def calculate_model_efficacy():
    """
    Calculate clustering model efficacy using Silhouette Score.
    
    Returns:
    - silhouette_score: Measure of how similar an object is to its own cluster compared to other clusters
    """
    # Calculate Silhouette Score
    score = silhouette_score(X, df['cluster'])
    return score

# Example: Find closest users to the user at index 0
closest_users_indices = find_closest_users(user_index=0, n_neighbors=5)

# Calculate model efficacy
model_efficacy = calculate_model_efficacy()

# Generate HTML content
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
        <p><strong>model Score:</strong> {:.2f}</p>
    </div>
</body>
</html>'''.format(model_efficacy)

# Save HTML to file
with open('closest_users.html', 'w') as file:
    file.write(html_content)

print("HTML file 'closest_users.html' has been created.")
