# id_card_detector.py
import tensorflow as tf
import numpy as np
import cv2

class IDCardDetector:
    def __init__(self, model_path='models/id_card_model.h5', threshold=0.5):
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold  # Set the threshold as an attribute

    def predict(self, frame):
        # Preprocess the frame
        processed_frame = self.preprocess_frame(frame)
        prediction = self.model.predict(processed_frame)
        
        # Return True if ID card detected, else False based on threshold
        return prediction[0][0] > self.threshold  # Adjusted for binary output

    def preprocess_frame(self, frame):
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        return np.expand_dims(normalized_frame, axis=0)
