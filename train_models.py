import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
from utils import extract_features  # Ensure utils.py exists

# Create models/ directory if missing
os.makedirs("models", exist_ok=True)

# Load metadata (update path as needed)
csv_path = os.path.join("datasets", "HAM10000", "HAM10000_metadata.csv")
metadata = pd.read_csv(csv_path)

# Simulate feature extraction (replace with your actual data loading)
# In practice, you'd loop through images and extract features
X_train = np.random.rand(100, 30)  # 100 samples, 30 features (mock data)
y_train = np.random.randint(0, 7, 100)  # Mock labels (0-6)

# Initialize and fit scaler
scaler = StandardScaler()
scaler.fit(X_train)  # Fit to training data
joblib.dump(scaler, "models/scaler.pkl")  # Save scaler

# Train and save models
models = {
    "svm": SVC(probability=True),
    "knn": KNeighborsClassifier(),
    "decision_tree": DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/{name}_classifier.pkl")

# Save label encoder (if needed)
le = LabelEncoder()
le.fit(y_train)
joblib.dump(le, "models/label_encoder.pkl")

print("✅ All models and scaler saved to models/")