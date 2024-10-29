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
└── id_card_detector.py         # ID card detector class
```

## 1.Dataset Preparation

Organize your dataset in the following folder structure:

datasets/with_id: Add images where people are wearing ID cards.
datasets/without_id: Add images where people are not wearing ID cards.
Make sure that each image is clear and standardized as much as possible for optimal model performance.

## 2.Training the Model
  **Run the train_and_evaluate.py script to train the model:**
  ```bash
  python train_and_evaluate.py
```
  **This script will:**

    1.Load and preprocess the images.
    2.Split the data into training and testing sets.
    3.Train a Convolutional Neural Network (CNN) model to classify "with ID" and "without ID" images.
    4.Save the trained model as id_card_detection_model.h5 in the models/ directory.

  **Training Script Breakdown (train_and_evaluate.py)**
  
    1.Load Images: load_images_and_labels() reads images from the dataset folders, resizing and normalizing them for the model.
    2.Model Definition: create_model() defines a CNN model suitable for binary classification.
    3.Metrics Calculation: After training, metrics such as accuracy, precision, recall, and F1-score are printed for evaluation.
    4.Model Saving: The trained model is saved in the models directory for future use in detection.

## 3.Running the Real-Time Detection Application
Once the model is trained, run main.py to start the real-time ID card detection:
  ```bash
  streamlit run main.py
```
This will launch a Streamlit application, allowing you to access the detection system through a web browser interface.

## 4.ID Card Detection Module
```bash
id_card_detector.py
```
this file contains the IDCardDetector class that handles loading the model, preprocessing frames, and predicting the presence of an ID card based on an input frame.

**Helper Functions**
```bash
utils/helper_functions.py
```
this contains reusable functions like draw_alert_message() to overlay alert messages on video frames.

## 5.Additional Notes
Adjust Threshold: You can adjust the threshold for ID card detection accuracy in id_card_detector.py.
Audio Alert: Ensure alert.mp3 is in the same directory as main.py to enable the alert sound.

**Contributing**
Contributions to improve this project are welcome. Feel free to submit issues and pull requests.

## License
This project is licensed under the MIT License.


