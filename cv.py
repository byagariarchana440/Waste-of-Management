import cv2
import numpy as np

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    image = cv2.resize(frame, (640, 480))

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Baby pink HSV range (adjust if needed)
    lower_pink = np.array([160, 50, 150])
    upper_pink = np.array([180, 150, 255])

    # Create a mask for baby pink
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Apply mask to original image
    result = cv2.bitwise_and(image, image, mask=mask)

    # Show the result
    cv2.imshow("Baby Pink Detection", result)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()