# import cv2
# import mediapipe as mp
# import numpy as np
# import os
#
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
#                                   min_tracking_confidence=0.5)
#
# def calculate_angle(p1, p2, p3):
#     v1 = np.array(p1) - np.array(p2)
#     v2 = np.array(p3) - np.array(p2)
#     v1_norm = np.linalg.norm(v1)
#     v2_norm = np.linalg.norm(v2)
#     if v1_norm == 0 or v2_norm == 0:
#         return 0.0
#     dot_product = np.dot(v1, v2)
#     cos_angle = dot_product / (v1_norm * v2_norm)
#     cos_angle = np.clip(cos_angle, -1.0, 1.0)
#     angle = np.arccos(cos_angle)
#     return np.degrees(angle)
#
# def get_landmark_point(landmarks, index, image_shape):
#     h, w = image_shape[:2]
#     return (int(landmarks[index].x * w), int(landmarks[index].y * h))
#
# def draw_landmark_and_vector(image, point, color):
#     cv2.circle(image, point, 3, color, -1)
#
# def draw_angle(image, p1, p2, p3, color, text):
#     cv2.line(image, p1, p2, color, 2)
#     cv2.line(image, p2, p3, color, 2)
#     angle = calculate_angle(p1, p2, p3)
#     mid_point = ((p1[0] + p3[0]) // 2, (p1[1] + p3[1]) // 2)
#     cv2.putText(image, f"{text}: {angle:.1f}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#
# cap = cv2.VideoCapture("emotions.mp4")
#
# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         break
#
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(image)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             landmarks = face_landmarks.landmark
#
#             # Önemli yüz nokalarını al
#             left_eye = get_landmark_point(landmarks, 33, image.shape)
#             right_eye = get_landmark_point(landmarks, 263, image.shape)
#             nose_tip = get_landmark_point(landmarks, 1, image.shape)
#             left_mouth = get_landmark_point(landmarks, 61, image.shape)
#             right_mouth = get_landmark_point(landmarks, 291, image.shape)
#             left_eyebrow = get_landmark_point(landmarks, 107, image.shape)
#             right_eyebrow = get_landmark_point(landmarks, 336, image.shape)
#
#             # Yeni eklenen dudak noktaları
#             upper_lip_top = get_landmark_point(landmarks, 0, image.shape)
#             upper_lip_bottom = get_landmark_point(landmarks, 13, image.shape)
#             lower_lip_top = get_landmark_point(landmarks, 14, image.shape)
#             lower_lip_bottom = get_landmark_point(landmarks, 17, image.shape)
#
#             # Tüm noktaları çiz
#             draw_landmark_and_vector(image, left_eye, (255, 0, 0))
#             draw_landmark_and_vector(image, right_eye, (255, 0, 0))
#             draw_landmark_and_vector(image, nose_tip, (0, 255, 0))
#             draw_landmark_and_vector(image, left_mouth, (0, 0, 255))
#             draw_landmark_and_vector(image, right_mouth, (0, 0, 255))
#             draw_landmark_and_vector(image, left_eyebrow, (255, 255, 0))
#             draw_landmark_and_vector(image, right_eyebrow, (255, 255, 0))
#
#             # Yeni eklenen dudak noktalarını çiz
#             draw_landmark_and_vector(image, upper_lip_top, (0, 255, 255))
#             draw_landmark_and_vector(image, upper_lip_bottom, (0, 255, 255))
#             draw_landmark_and_vector(image, lower_lip_top, (0, 255, 255))
#             draw_landmark_and_vector(image, lower_lip_bottom, (0, 255, 255))
#
#             # Açıları çiz
#             draw_angle(image, left_mouth, nose_tip, right_mouth, (0, 255, 255), "Mouth")
#             draw_angle(image, left_eyebrow, nose_tip, right_eyebrow, (255, 0, 255), "Eyebrow")
#             draw_angle(image, left_eye, nose_tip, right_eye, (255, 255, 0), "Eye")
#
#             # Dudak noktalarının koordinatlarını ekranda göster
#             cv2.putText(image, f"Upper Lip Top: {upper_lip_top}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#             cv2.putText(image, f"Upper Lip Bottom: {upper_lip_bottom}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#             cv2.putText(image, f"Lower Lip Top: {lower_lip_top}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#             cv2.putText(image, f"Lower Lip Bottom: {lower_lip_bottom}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#
#     cv2.imshow('MediaPipe Face Mesh', image)
#     if cv2.waitKey(0) & 0xFF == ord("q"):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np

# Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)


def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
    dot_product = np.dot(v1, v2)
    cos_angle = dot_product / (v1_norm * v2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def get_landmark_point(landmarks, index, image_shape):
    h, w = image_shape[:2]
    return (int(landmarks[index].x * w), int(landmarks[index].y * h))


def draw_angle(image, p1, p2, p3, color, text):
    cv2.line(image, p1, p2, color, 2)
    cv2.line(image, p2, p3, color, 2)
    angle = calculate_angle(p1, p2, p3)
    mid_point = ((p1[0] + p3[0]) // 2, (p1[1] + p3[1]) // 2)
    cv2.putText(image, f"{text}: {angle:.1f}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return angle


def get_emotion(landmarks, image):
    # Yüz noktalarını al
    left_eye = get_landmark_point(landmarks, 33, image.shape)
    right_eye = get_landmark_point(landmarks, 263, image.shape)
    nose_tip = get_landmark_point(landmarks, 1, image.shape)
    left_mouth = get_landmark_point(landmarks, 61, image.shape)
    right_mouth = get_landmark_point(landmarks, 291, image.shape)
    upper_lip_top = get_landmark_point(landmarks, 0, image.shape)
    lower_lip_bottom = get_landmark_point(landmarks, 17, image.shape)

    # Gözler arasındaki mesafeyi hesapla
    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

    # Dudaklar arasındaki mesafeyi hesapla
    mouth_distance = np.linalg.norm(np.array(left_mouth) - np.array(right_mouth))

    # Dudakların konumuna göre bir özellik belirle
    mouth_height = np.linalg.norm(np.array(upper_lip_top) - np.array(lower_lip_bottom))

    # Ağız ve göz açılarını hesapla
    mouth_angle = calculate_angle(left_mouth, nose_tip, right_mouth)
    eyebrow_angle = calculate_angle(left_eye, nose_tip, right_eye)

    # Temel duygu durumlarını belirlemek için basit kurallar
    if mouth_height > 20 and mouth_distance > 50 and mouth_angle < 30:
        return "Happy"
    elif mouth_height < 10 and mouth_distance < 50 and eyebrow_angle > 15:
        return "Sad"
    elif mouth_height > 20 and mouth_distance > 50 and eyebrow_angle > 30:
        return "Surprised"
    else:
        return "Neutral"


cap = cv2.VideoCapture("emotions.mp4")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Yüz noktalarını al
            left_eye = get_landmark_point(landmarks, 33, image.shape)
            right_eye = get_landmark_point(landmarks, 263, image.shape)
            nose_tip = get_landmark_point(landmarks, 1, image.shape)
            left_mouth = get_landmark_point(landmarks, 61, image.shape)
            right_mouth = get_landmark_point(landmarks, 291, image.shape)
            upper_lip_top = get_landmark_point(landmarks, 0, image.shape)
            lower_lip_bottom = get_landmark_point(landmarks, 17, image.shape)

            # Açıları çiz
            draw_angle(image, left_eye, nose_tip, right_eye, (255, 255, 0), "Eye")
            draw_angle(image, left_mouth, nose_tip, right_mouth, (0, 255, 255), "Mouth")

            # Dudak noktalarının koordinatlarını ekranda göster
            cv2.putText(image, f"Upper Lip Top: {upper_lip_top}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            cv2.putText(image, f"Lower Lip Bottom: {lower_lip_bottom}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

            # Duygu durumunu belirle ve ekranda göster
            emotion = get_emotion(landmarks, image)
            cv2.putText(image, f"Emotion: {emotion}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if emotion == "Happy" else (0, 0, 255), 2)

    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
