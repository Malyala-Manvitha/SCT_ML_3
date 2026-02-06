import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog

# Dataset path
DATASET_PATH = "dataset"
CATEGORIES = ["cats", "dogs"]
IMAGE_SIZE = (64, 64)

features = []
labels = []

print("Loading and processing images...")

for label, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATASET_PATH, category)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMAGE_SIZE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            hog_features = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys'
            )

            features.append(hog_features)
            labels.append(label)

        except Exception as e:
            print(f"Error loading image {img_path}")

features = np.array(features)
labels = np.array(labels)

print("Dataset prepared successfully!")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

print("Training SVM model...")

model = SVC(kernel='linear')
model.fit(X_train, y_train)

print("Model training completed!")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))
