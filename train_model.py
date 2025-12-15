import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load dataset
df = pd.read_csv(os.path.expanduser("~/.cache/kagglehub/datasets/umeradnaan/tourism-dataset/versions/1/tourism_dataset.csv"))

# Select only the 4 features you want
X = df[["Visitors", "Location", "Country", "Category"]]
y = df["Revenue"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
joblib.dump(model, "tourism_rf_model.pkl")

print("✅ Model trained and saved as tourism_rf_model.pkl")