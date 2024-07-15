  import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
import pickle
import chardet

# Detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']

# Load data
def load_data(file_path):
    """Load product rating data from CSV with handling for encoding and errors."""
    encoding = detect_encoding(file_path)
    try:
        # Using 'on_bad_lines' to skip problematic lines
        return pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading file: {e}")
        raise

# Prepare data for Surprise
def prepare_data(df):
    """Prepare data for collaborative filtering using Surprise."""
    reader = Reader(rating_scale=(1, 5))  # Adjust the rating scale if necessary
    data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)
    return data

# Build and train model
def build_and_train_model(data):
    """Train a collaborative filtering model using SVD."""
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD()  # Singular Value Decomposition
    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    print(f'RMSE: {rmse}')
    return model

# Recommend products for a specific user
def recommend_products(model, user_id, all_product_ids, top_n=10):
    """Recommend top N products for a user based on model predictions."""
    predictions = []
    for product_id in all_product_ids:
        pred = model.predict(user_id, product_id)
        predictions.append((product_id, pred.est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]

# Save model to a file
def save_model(model, file_path):
    """Save the trained model to a file."""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

# Load model from a file
def load_model(file_path):
    """Load the trained model from a file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Main script
if __name__ == "__main__":
    # Specify the path to your CSV file
    file_path = 'product_ratings.csv'  # Update this path if needed
    
    # Load data
    df = load_data(file_path)
    
    # Prepare data for the recommendation model
    data = prepare_data(df)
    
    # Build and train the model
    model = build_and_train_model(data)
    
    # Save the trained model
    save_model(model, 'product_recommendation_model.pkl')
    
    # Example: Recommend products for a specific user
    user_id = 1  # Example user ID; change as needed
    all_product_ids = df['product_id'].unique()
    recommendations = recommend_products(model, user_id, all_product_ids)
    
    print('Top Recommendations:')
    for product_id, score in recommendations:
        print(f'Product ID: {product_id}, Predicted Rating: {score}')
    
    # Load the model (example usage)
    loaded_model = load_model('product_recommendation_model.pkl')
    
    # Recommend products using the loaded model
    recommendations_loaded_model = recommend_products(loaded_model, user_id, all_product_ids)
    
    print('Top Recommendations (Loaded Model):')
    for product_id, score in recommendations_loaded_model:
        print(f'Product ID: {product_id}, Predicted Rating: {score}')
