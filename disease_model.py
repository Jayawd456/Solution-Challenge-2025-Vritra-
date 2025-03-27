import os
import pickle

# Get the absolute path of the current directory
model_path = os.path.join(os.path.dirname(__file__), 'disease_model.pkl')

# Check if the file exists before loading
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")
