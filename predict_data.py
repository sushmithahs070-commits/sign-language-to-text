import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the trained model and labels
try:
    model, labels = joblib.load('sign_language_model.pkl')
    print("Model and labels loaded successfully.")
    print("Labels:", labels)
    print("Model:", model)
except FileNotFoundError:
    print("Error: sign_language_model.pkl not found. Make sure it's in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
print("Webcam initialized.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 3D landmark data
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            data = np.array(data).reshape(1, -1)

            try:
                prediction = model.predict(data)[0]
                confidence = np.max(model.predict_proba(data))

                predicted_label = labels[prediction] if prediction < len(labels) else "Unknown"

                cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print(f"Prediction error during loop: {e}")
                cv2.putText(frame, "Prediction Error", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()