import cv2

def draw_alert_message(frame, message, color=(0, 0, 255)):
    """Draws an alert message on the frame with the specified color."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    position = (50, 50)  # Position of the message

    # Draw text on the frame
    cv2.putText(frame, message, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame
