import pickle
import os

# Path to the model file
model_path = os.path.expanduser("~/Downloads/random_forest_model.pkl")

def load_model():
    """Load the random forest model from the specified path."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Example usage
if __name__ == "__main__":
    model = load_model()
    print("Model has been loaded successfully.")
    print(f"Model type: {type(model)}")
    
    # Sample input data for prediction
    sample_input = [[0.5, 1.5, 2.5, 3.5]]  # Adjust this based on your model's expected input shape
    # Check if the model is a valid estimator before making predictions
    if hasattr(model, 'predict'):
        predictions = model.predict(sample_input)
    else:
        predictions = "Model is not a valid estimator."

    print(f"Predictions: {predictions}")

    # Optionally, print model parameters or other relevant information
