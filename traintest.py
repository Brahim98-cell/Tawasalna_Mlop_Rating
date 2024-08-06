import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# Load the data
def load_data():
    data = pd.read_csv('TawasalnaDB.product.csv')
    data = data.drop(columns=['image', '_class'])
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
    categorical_features = encoder.fit_transform(data[['productCategory']]).toarray()

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
    similar_products = [score[0] for score in similar_scores[1:top_n+1]]
    return similar_products

# Function to generate HTML file
def generate_html(products, num_features, num_products):
    products_html = "".join(f"""
    <div class="product">
        <h2>{product['title']}</h2>
        <p><strong>Description:</strong> {product['description']}</p>
        <p><strong>Price:</strong> ${product['price']}</p>
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
product_id = '6654f305cbd64a007f1ac9cf'  # Replace with the product ID you want to test

if product_id in data['_id'].values:
    product_index = data.index[data['_id'] == product_id].tolist()[0]
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
