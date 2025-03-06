from stored_dataset.store_dataset import load_model

# Test loading the model
model = load_model()
print(f"Loaded model type: {type(model)}")
