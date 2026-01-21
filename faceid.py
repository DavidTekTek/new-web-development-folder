import cv2
import face_recognition
import numpy as np
import os

DATA_FILE = "face_id_user.npy"   # variable storing where face data is saved


def capture_face_encoding(prompt="Look at the camera and press SPACE to capture..."):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not access the camera.")
        return None

    print(prompt)
    encoding = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Face ID - Press SPACE to capture, ESC to cancel", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key == 32:  # SPACE
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)

            if len(encodings) == 0:
                print("No face detected. Try again with better lighting.")
            elif len(encodings) > 1:
                print("Multiple faces detected. Please show only one face.")
            else:
                encoding = encodings[0]
                print("Face captured!")
                break

    cap.release()
    cv2.destroyAllWindows()
    return encoding


def enroll_user():
    user_face_encoding = capture_face_encoding("ENROLL: Look at the camera and press SPACE...")
    if user_face_encoding is None:
        print("Enrollment canceled/failed.")
        return

    np.save(DATA_FILE, user_face_encoding)
    print(f"Enrollment successful! Face ID saved to '{DATA_FILE}'.")


def sign_in_with_face_id(tolerance=0.50):
    if not os.path.exists(DATA_FILE):
        print("No enrolled user found. Please enroll first.")
        return

    stored_encoding = np.load(DATA_FILE)

    live_encoding = capture_face_encoding("SIGN IN: Look at the camera and press SPACE...")
    if live_encoding is None:
        print("Sign-in canceled/failed.")
        return

    distance = face_recognition.face_distance([stored_encoding], live_encoding)[0]
    is_match = distance <= tolerance

    if is_match:
        print("✅ Face ID verified. Sign-in successful!")
    else:
        print("❌ Face did not match. Access denied.")
        print(f"(match score distance: {distance:.3f}, tolerance: {tolerance})")


def main():
    while True:
        print("\n--- Face ID Sign-In App ---")
        print("1) Enroll (register face)")
        print("2) Sign in with Face ID")
        print("3) Exit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            enroll_user()
        elif choice == "2":
            sign_in_with_face_id()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()
