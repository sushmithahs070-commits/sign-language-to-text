import cv2
import os
import numpy as np
import mediapipe as mp
import time
from playsound import playsound
import threading

# Optional: Function to play beep sound asynchronously
def play_beep():
    threading.Thread(target=playsound, args=('beep.wav',), daemon=True).start()

# Ask for gesture label
label = input("Enter the label for this gesture (e.g., A, B, Hello): ")

# Create folder
DATA_DIR = 'data'
label_dir = os.path.join(DATA_DIR, label)
os.makedirs(label_dir, exist_ok=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Countdown before capture starts
print("Get ready! Capturing will start in:")
for i in range(3, 0, -1):
    print(i)
    time.sleep(1)

print("Capture started. Press SPACE to save data, 'q' to quit.")

counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show info on screen
    cv2.putText(frame, f"Label: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Samples Collected: {counter}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == 32 and result.multi_hand_landmarks:
        # Extract landmarks
        landmarks = []
        for lm in result.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks)
        np.save(os.path.join(label_dir, f"{counter}.npy"), landmarks)
        counter += 1
        play_beep()  # Play beep on save

cap.release()
cv2.destroyAllWindows()
