

import cv2
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace


def draw_label(frame, label, left, top):
    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# Initialize the video capture and MTCNN detector
cap = cv2.VideoCapture("C:/Users/jesus/PycharmProjects/duyguanaliz/Emotion_Detection_CNN/emotions_human.mp4")
detector = MTCNN()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using MTCNN
    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']

        # Extract the face from the frame
        face_img = frame[y:y + h, x:x + w]

        # Predict the emotion using DeepFace
        try:
            analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw the emotion label
            draw_label(frame, emotion, x, y)
        except Exception as e:
            print(f"Error in emotion detection: {e}")

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
