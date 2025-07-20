

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv("gld_price_data.csv")  # Put the CSV in same folder

# Prepare features and target
X = df.drop(['Date', 'GLD'], axis=1)
y = df['GLD']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open("gold_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model retrained and saved successfully.")
