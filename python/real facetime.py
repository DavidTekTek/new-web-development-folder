import cv2
import sys

# Load pre-trained Haar Cascade Classifier for face detection
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Verify the cascade loaded properly
if face_cascade.empty():
    print(f"Error: Could not load Haar cascade from: {cascade_path}")
    sys.exit(1)

# Initialize video capture (use webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

# (Optional) Set camera resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture image.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # (Optional) Improve detection reliability a bit
    gray = cv2.equalizeHist(gray)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the count of faces
    cv2.putText(
        frame,
        f"People Count: {len(faces)}",
        (10, 30),
        font,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )

    # Display the frame with face detection and people count
    cv2.imshow("Face Tracking and Counting", frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

