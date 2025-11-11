import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib  # Use joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
X = []
y = []

# Load all the data
for label, folder in enumerate(os.listdir(DATA_DIR)):

    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            data = np.load(os.path.join(folder_path, file))
            X.append(data)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model Accuracy:", model.score(X_test, y_test))

# Save model using joblib
joblib.dump((model, os.listdir(DATA_DIR)), 'sign_language_model.pkl')  # Save model with joblib
