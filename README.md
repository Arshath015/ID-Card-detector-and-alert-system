# ID Card Detection System

This project is an ID Card Detection System built with TensorFlow, Keras, and OpenCV, utilizing Streamlit for a simple user interface. It classifies real-time video frames into "with ID" and "without ID" categories and plays an alert sound when an ID card is not detected.

## Project Structure

```plaintext
id-card-detection
├── datasets                    # Directory containing training data
│   ├── with_id                 # Images of people wearing ID cards
│   ├── without_id              # Images of people without ID cards
├── models                      # Pre-trained model folder
│   └── id_card_detection_model.h5   # Saved ID card detection model
├── utils
│   └── helper_functions.py     # Helper functions, such as drawing alert messages on frames
├── main.py                     # Main Streamlit application file
├── train_and_evaluate.py       # Script for training and evaluating the model
├── id_card_detector.py         # ID card detector class
└── requirements.txt            # Dependencies required to run the project
```

## Dataset Preparation

Organize your dataset in the following folder structure:

datasets/with_id: Add images where people are wearing ID cards.
datasets/without_id: Add images where people are not wearing ID cards.
Make sure that each image is clear and standardized as much as possible for optimal model performance.
