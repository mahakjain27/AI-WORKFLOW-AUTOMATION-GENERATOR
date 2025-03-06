from flask import Flask, render_template, request
from stored_dataset.store_dataset import load_model, create_label_encoders
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    model = load_model()  
    if not hasattr(model, 'predict'):
        raise ValueError("Loaded model is not a valid estimator.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set model to None if loading fails

# Create label encoders once (avoid redundant calls)
label_encoders = create_label_encoders()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text='Error: Model failed to load.')

    # Define input fields
    input_fields = [
        'business_type', 'workflow_automation_idea', 'workflow_type', 'key_objective',
        'current_challenges', 'desired_features', 'existing_tools', 'preferred_integrations', 'data_sources'
    ]

    # Extract and clean input data
    input_data = []
    for field in input_fields:
        value = request.form.get(field, '').replace('"', '').strip()
        if not value:
            return render_template('index.html', prediction_text='Error: All input fields must be filled out.')

        # Print the value being processed
        print(f"Processing {field}: {value}")

        # Encode categorical values if applicable
        if field in label_encoders.keys():
            try:
                transformed_value = label_encoders[field].transform([value])[0]
            except ValueError:
                transformed_value = 0  # Default for unseen labels
        else:
            try:
                transformed_value = float(value)  # Convert to float for numerical fields
            except ValueError:
                transformed_value = 0  # Default for invalid input

        input_data.append(transformed_value)

    # Ensure correct feature size
    while len(input_data) < len(input_fields):  
        input_data.append(0)  # Pad missing features with default values

    input_array = np.array(input_data).reshape(1, -1)

    # Debugging prints
    print("Final Input Data for Prediction:", input_array)
    for field in input_fields:
        if field in label_encoders.keys():
            print(f"{field} classes: {label_encoders[field].classes_}")

    try:
        # Make prediction
        prediction = model.predict(input_array)
        prediction_result = prediction[0] if prediction[0] != 0 else "No valid suggestion available."
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error during prediction: {str(e)}')

    return render_template('index.html', prediction_text=f'Predicted Workflow Suggestion: {prediction_result}')

if __name__ == "__main__":
    app.run(debug=True)
