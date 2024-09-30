import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

# Load the data
def load_data():
    url = 'https://business.tawasalna.com/tawasalna-business/products/getProducts/active'
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    
    data = pd.json_normalize(response.json())
    
    # Drop columns that do not need to be used
    columns_to_drop = [
        'publisher.id', 'publisher.email', 'publisher.community.id',
        'publisher.community.name', 'publisher.community.description', 
        'publisher.name', 'productCategory.id', 'productCategory.description', 
        'productCategory.cover', 'productCategory.isActive', 
        'productCategory.productsCount'
    ]
    
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    
    data = data.drop(columns=existing_columns_to_drop)
    data['averageStars'] = data['averageStars'].fillna(data['averageStars'].mean())
    return data

# Prepare features
def prepare_features(data):
    tfidf_title = TfidfVectorizer(stop_words='english')
    tfidf_description = TfidfVectorizer(stop_words='english')

    title_features = tfidf_title.fit_transform(data['title'])
    description_features = tfidf_description.fit_transform(data['description'])

    numerical_features = data[['price', 'totalReviews', 'averageStars']]
    scaler = StandardScaler()
    scaled_numerical_features = scaler.fit_transform(numerical_features)

    encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_features = encoder.fit_transform(data[['productCategory.name']]).toarray()

    combined_features = hstack([title_features, description_features, scaled_numerical_features, categorical_features])
    
    return combined_features

# Compute similarity matrix
def compute_similarity(features):
    return cosine_similarity(features)

# Recommendation function
def get_similar_products(product_index, top_n=5):
    similar_scores = list(enumerate(similarity_matrix[product_index]))
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
    similar_products = [score[0] for score in similar_scores[1:top_n + 1]]
    return similar_products

# HTML template for rendering
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Product Recommendations</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }
        .container { max-width: 800px; margin: auto; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }
        .product { border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px; background: #f9f9f9; }
        .product h2 { margin: 0; font-size: 1.5em; }
        .product p { margin: 5px 0; }
        img { border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Top Recommended Products</h1>
        {% for product in products %}
        <div class="product">
            <h2>{{ product['title'] }}</h2>
            <p><strong>Description:</strong> {{ product['description'] }}</p>
            <p><strong>Price:</strong> ${{ product['price'] }}</p>
            <img src="https://upload.tawasalna.com/{{ product['image'] }}" alt="{{ product['title'] }}" style="width:150px; height:auto;" />
        </div>
        {% endfor %}
        <h2>Model Effectiveness</h2>
        <p>Total number of features used: {{ num_features }}</p>
        <p>Total number of products in the dataset: {{ num_products }}</p>
    </div>
</body>
</html>
"""

# Initialize data and features
data = load_data()
features = prepare_features(data)
similarity_matrix = compute_similarity(features)

# API endpoint for recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    product_id = '663d93b81d70010259349716'  # Hardcoded product ID for comparison
    if product_id in data['id'].values:
        product_index = data.index[data['id'] == product_id].tolist()[0]
        similar_products_indices = get_similar_products(product_index, top_n=5)
        
        if similar_products_indices:
            recommended_products = data.iloc[similar_products_indices]
            num_features = features.shape[1]
            num_products = len(data)

            # Render HTML template with product recommendations
            return render_template_string(HTML_TEMPLATE, products=recommended_products.to_dict(orient='records'), num_features=num_features, num_products=num_products)
        else:
            return jsonify({"message": "No recommendations available."}), 404
    else:
        return jsonify({"message": "Product ID not found."}), 404

if __name__ == '__main__':
    app.run(host='185.192.96.18', port=5004)
