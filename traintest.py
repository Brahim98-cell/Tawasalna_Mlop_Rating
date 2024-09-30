import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from PIL import Image
import matplotlib.pyplot as plt

# Load the data
def load_data():
    url = 'https://business.tawasalna.com/tawasalna-business/products/getProducts/active'
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    
    data = pd.json_normalize(response.json())
    
    # Print the first few rows and the columns to understand the structure
    print(data.head())
    print(data.columns)
    
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

# Initialize data and features
data = load_data()
features = prepare_features(data)
similarity_matrix = compute_similarity(features)

# Recommendation function
def get_similar_products(product_index, top_n=5):
    similar_scores = list(enumerate(similarity_matrix[product_index]))
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
    similar_products = [score[0] for score in similar_scores[1:top_n + 1]]
    return similar_products

# Function to generate HTML file
def generate_html(products, num_features, num_products):
    base_image_url = "https://upload.tawasalna.com/"  # Base URL for images
    products_html = "".join(f"""
    <div class="product">
        <h2>{product['title']}</h2>
        <p><strong>Description:</strong> {product['description']}</p>
        <p><strong>Price:</strong> ${product['price']}</p>
        <img src="{base_image_url}{product['image']}" alt="{product['title']}" style="width:150px; height:auto;" />
    </div>
    """ for product in products)

    html_content = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Product Recommendations</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .product {{ border: 1px solid #ddd; padding: 10px; margin: 10px 0; }}
            .product h2 {{ margin: 0; }}
            .product p {{ margin: 5px 0; }}
        </style>
    </head>
    <body>
        <h1>Top Recommended Products</h1>
        {products_html}
        <h2>Model Effectiveness</h2>
        <p>Total number of features used: {num_features}</p>
        <p>Total number of products in the dataset: {num_products}</p>
    </body>
    </html>
    """
    
    with open('recommendation.html', 'w') as file:
        file.write(html_content)



# Example usage
product_id = '663d93b81d70010259349716'  # Replace with the product ID you want to test

if product_id in data['id'].values:
    product_index = data.index[data['id'] == product_id].tolist()[0]
    similar_products_indices = get_similar_products(product_index, top_n=5)
    
    if similar_products_indices:
        recommended_products = data.iloc[similar_products_indices]
        num_features = features.shape[1]
        num_products = len(data)

        # Print to console
        print("Top Recommended Products:")
        for _, product in recommended_products.iterrows():
            print(f"Title: {product['title']}")
            print(f"Description: {product['description']}")
            print(f"Price: ${product['price']}")
            print()

        # Generate HTML file
        generate_html(recommended_products.to_dict(orient='records'), num_features, num_products)
        print("HTML file 'recommendation.html' has been generated.")
    else:
        print("No recommendations available.")
else:
    print("Product ID not found.")
