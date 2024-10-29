import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Paths to dataset folders
with_id_path = 'datasets/with_id'
without_id_path = 'datasets/without_id'

# Parameters
img_size = (128, 128)
test_size = 0.2  # 20% for testing

# Load images and labels
def load_images_and_labels():
    images, labels = [], []
    for label, path in enumerate([with_id_path, without_id_path]):  # Label 0 for with_id, 1 for without_id
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load and preprocess data
images, labels = load_images_and_labels()
images = images / 255.0  # Normalize
labels = to_categorical(labels, 2)  # Convert labels to categorical for binary classification

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)

# Model definition
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model
model = create_model()
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=32)

# Evaluate model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_test_classes, y_pred_classes)
precision = precision_score(y_test_classes, y_pred_classes, average='binary')
recall = recall_score(y_test_classes, y_pred_classes, average='binary')
f1 = f1_score(y_test_classes, y_pred_classes, average='binary')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Save the model for real-time detection
model.save("id_card_detection_model.h5")
