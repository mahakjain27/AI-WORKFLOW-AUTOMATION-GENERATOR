import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_excel("C:/Users/DELL/Downloads/FINAL_Cleaned.xlsx")

# Preprocess the data
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split the data into features and target
print("Column names in the dataset:", df.columns)  # Debugging line to print column names
X = df.drop('workflow_suggestion', axis=1)  # Updated to use the actual target column name


y = df['workflow_suggestion']  # Updated to use the actual target column name


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open("C:/Users/DELL/Downloads/random_forest_model.pkl", 'wb') as file:
    pickle.dump(model, file)

print("Model has been trained and saved successfully.")
