import cv2
import numpy as np

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Define color range for the hand (Green color - You may need to adjust these values)
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to segment the hand color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw a rectangle around the detected area
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detect thumbs up and thumbs down based on the bounding box dimensions
        if h > 2 * w:
            cv2.putText(frame, "Thumbs Down", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif w > 2 * h:
            cv2.putText(frame, "Thumbs Up", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Thumbs Up/Down Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
