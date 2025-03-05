from flask import Flask, render_template, request
from stored_dataset.store_dataset import load_model


import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model()



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    input_data = [float(x) for x in request.form.values()]
    input_data = np.array(input_data).reshape(1, -1)

    try:
        # Make prediction
        prediction = model.predict(input_data)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


    return render_template('index.html', prediction_text=f'Predicted Workflow Suggestion: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
