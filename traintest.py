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
    return model, rmse

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

# Convert recommendations to HTML
def recommendations_to_html(recommendations, file_path):
    """Convert the recommendations to an HTML file."""
    df = pd.DataFrame(recommendations, columns=['Product ID', 'Predicted Rating'])
    html_content = df.to_html(index=False)
    with open(file_path, 'w') as f:
        f.write(html_content)

# Generate HTML report
def generate_html_report(rmse, recommendations, file_path):
    """Generate an HTML report for the model evaluation."""
    with open(file_path, 'w') as f:
        f.write('<html><body>')
        f.write('<h1>Recommendation Model Evaluation Report</h1>')
        f.write('<p>RMSE of the SVD model is: {:.2f}</p>'.format(rmse))
        f.write('<h2>Top Recommendations</h2>')
        
        # Convert recommendations to HTML and include in the report
        recommendations_df = pd.DataFrame(recommendations, columns=['Product ID', 'Predicted Rating'])
        recommendations_html = recommendations_df.to_html(index=False)
        f.write(recommendations_html)
        
        f.write('</body></html>')

# Main script
if __name__ == "__main__":
    # Specify the path to your CSV file
    file_path = 'product_ratings.csv'  # Update this path if needed
    
    # Load data
    df = load_data(file_path)
    
    # Prepare data for the recommendation model
    data = prepare_data(df)
    
    # Build and train the model
    model, rmse = build_and_train_model(data)
    
    # Save the trained model
    save_model(model, 'product_recommendation_model.pkl')
    
    # Example: Recommend products for a specific user
    user_id = 1  # Example user ID; change as needed
    all_product_ids = df['product_id'].unique()
    recommendations = recommend_products(model, user_id, all_product_ids)
    
    # Convert and save recommendations to HTML
    recommendations_to_html(recommendations, 'recommendations.html')
    
    # Generate HTML report
    generate_html_report(rmse, recommendations, 'model_evaluation_report.html')
    
    # Load the model (example usage)
    loaded_model = load_model('product_recommendation_model.pkl')
    
    # Recommend products using the loaded model
    recommendations_loaded_model = recommend_products(loaded_model, user_id, all_product_ids)
    
    # Convert and save recommendations from the loaded model to HTML
    recommendations_to_html(recommendations_loaded_model, 'recommendations_loaded.html')
    
    # Generate HTML report for loaded model
    generate_html_report(rmse, recommendations_loaded_model, 'model_evaluation_report_loaded.html')
