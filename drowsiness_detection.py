import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import pygame
pygame.mixer.init()
def play_alarm():
    pygame.mixer.music.load("alarm.mp3")
    pygame.mixer.music.play()
# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmark indices (MediaPipe)
LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

# EAR threshold
EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20

counter = 0

# Function to calculate Eye Aspect Ratio
def eye_aspect_ratio(eye_points, landmarks, w, h):
    
    coords = []

    for point in eye_points:
        x = int(landmarks[point].x * w)
        y = int(landmarks[point].y * h)
        coords.append((x,y))

    A = distance.euclidean(coords[1], coords[5])
    B = distance.euclidean(coords[2], coords[4])
    C = distance.euclidean(coords[0], coords[3])

    ear = (A + B) / (2.0 * C)
    return ear


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    h,w,_ = frame.shape

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            landmarks = face_landmarks.landmark

            leftEAR = eye_aspect_ratio(LEFT_EYE, landmarks, w, h)
            rightEAR = eye_aspect_ratio(RIGHT_EYE, landmarks, w, h)

            ear = (leftEAR + rightEAR) / 2.0

            cv2.putText(frame, f"EAR: {ear:.2f}", (30,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

            if ear < EAR_THRESHOLD:
                counter += 1

                if counter == FRAME_THRESHOLD:
                    cv2.putText(frame, "DROWSINESS ALERT!",
                                (150,200),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (0,0,255),
                                3)
                    play_alarm()

            else:
                counter = 0

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()