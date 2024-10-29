# main.py
import cv2
from keras.models import load_model
from utils.helper_functions import draw_alert_message
from playsound import playsound
import streamlit as st
import threading
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("id_card_detection_model.h5")

def play_alert():
    """Plays the alert sound in a separate thread to avoid blocking."""
    playsound('alert.mp3')

def detect_id_card(frame):
    """Runs the ID card detection model on the frame."""
    img = cv2.resize(frame, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return np.argmax(prediction) == 0  # Return True if "with ID" is predicted

def main():
    st.title("ID Card Detection System")
    st.text("Press 'Stop' to end the application.")

    # Initialize Streamlit to stop the loop with a button
    stop_button = st.button("Stop")

    cap = cv2.VideoCapture(0)
    alert_played = False

    # Streamlit video display
    video_display = st.image([])

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect ID card presence
        id_card_detected = detect_id_card(frame)

        if not id_card_detected:
            frame = draw_alert_message(frame, "Please wear ID card", color=(0, 0, 255))  # Red alert message
            if not alert_played:
                threading.Thread(target=play_alert).start()
                alert_played = True
            status_message = "Not Wearing ID Card"
        else:
            alert_played = False
            frame = draw_alert_message(frame, "Thank You!", color=(0, 255, 0))  # Green thank you message
            status_message = "Wearing ID Card"

        # Convert the frame color for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_display.image(frame_rgb)
        
        # Display the status message in Streamlit
        st.text(status_message)

    cap.release()

if __name__ == "__main__":
    main()
