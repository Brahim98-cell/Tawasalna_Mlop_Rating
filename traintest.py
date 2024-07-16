import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import pickle

# Load data
def load_data():
    """Load resident data from CSV."""
    return pd.read_csv('resident_data.csv')

# Prepare data
def prepare_data(df):
    """Prepare data by selecting features, encoding categorical variables, and scaling."""
    # Encode categorical features
    label_encoder_occupation = LabelEncoder()
    label_encoder_community = LabelEncoder()
    
    df['occupation_encoded'] = label_encoder_occupation.fit_transform(df['occupation'])
    df['community_encoded'] = label_encoder_community.fit_transform(df['community'])
    
    # Select features
    features = df[['age', 'income', 'occupation_encoded', 'community_encoded']]
    
    return features, label_encoder_occupation, label_encoder_community

# Train and match model
def train_and_match_model(features):
    """Train a K-Nearest Neighbors model and perform matching."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    model = NearestNeighbors(n_neighbors=5, algorithm='auto')
    model.fit(scaled_features)
    
    return model, scaler

# Match residents
def match_residents(model, scaler, occupation_encoder, community_encoder, new_resident_features):
    """Match a new resident with existing residents."""
    # Convert categorical features to numeric
    new_resident_features_encoded = new_resident_features.copy()
    new_resident_features_encoded[2] = occupation_encoder.transform([new_resident_features[2]])[0]
    new_resident_features_encoded[3] = community_encoder.transform([new_resident_features[3]])[0]
    
    # Scale features
    scaled_new_resident = scaler.transform([new_resident_features_encoded])
    
    # Find nearest neighbors
    distances, indices = model.kneighbors(scaled_new_resident)
    return indices

# Save model
def save_model(model, scaler, occupation_encoder, community_encoder):
    """Save the trained model, scaler, and encoders."""
    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('occupation_encoder.pkl', 'wb') as f:
        pickle.dump(occupation_encoder, f)
    with open('community_encoder.pkl', 'wb') as f:
        pickle.dump(community_encoder, f)

# Load model
def load_model():
    """Load the trained model, scaler, and encoders."""
    with open('knn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('occupation_encoder.pkl', 'rb') as f:
        occupation_encoder = pickle.load(f)
    with open('community_encoder.pkl', 'rb') as f:
        community_encoder = pickle.load(f)
    return model, scaler, occupation_encoder, community_encoder

# Main script
if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Prepare data
    features, occupation_encoder, community_encoder = prepare_data(df)
    
    # Train and match model
    model, scaler = train_and_match_model(features)
    
    # Save the model, scaler, and encoders
    save_model(model, scaler, occupation_encoder, community_encoder)
    
    # Example: Match a new resident
    new_resident_features = [30, 50000, 'Teacher', 'Downtown']  # Example new resident features
    indices = match_residents(model, scaler, occupation_encoder, community_encoder, new_resident_features)
    
    print(f'Matching residents indices: {indices}')
    
    # Load the model (example usage)
    loaded_model, loaded_scaler, loaded_occupation_encoder, loaded_community_encoder = load_model()
    
    # Match a new resident using the loaded model
    indices_loaded_model = match_residents(loaded_model, loaded_scaler, loaded_occupation_encoder, loaded_community_encoder, new_resident_features)
    
    print(f'Matching residents indices (Loaded Model): {indices_loaded_model}')
