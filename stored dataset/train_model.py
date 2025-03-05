import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_excel("C:/Users/DELL/Downloads/FINAL_Cleaned.xlsx")


# Encode categorical features
label_encoders = {}
encoded_df = df.copy()

for column in df.columns:
    le = LabelEncoder()
    encoded_df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split data into features (X) and target (y)
X = encoded_df.drop(columns=['workflow_suggestion'])
y = encoded_df['workflow_suggestion']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save trained model
joblib.dump(model, "C:/Users/DELL/Downloads/random_forest_model.pkl")
