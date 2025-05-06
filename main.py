import cv2
import mediapipe as mp
import math
import time
from desktop_notifier import DesktopNotifier
import asyncio

notifier = DesktopNotifier()

mouth_landmarks = [0, 267, 269, 270, 409, 306, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]
fingertip_landmarks = [4, 8, 12, 16, 20]

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

mp_drawing = mp.solutions.drawing_utils
webcam = cv2.VideoCapture(0)

hand_touch_start = None
face_touch_start = None

async def sendNotif(message):
    await notifier.send(title='HANDS APART!', message=message)

def calculate_dist(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def hands_touching():
    left_index_finger = left_hand[8]
    right_index_finger = right_hand[8]
    dist = calculate_dist(left_index_finger, right_index_finger)
    return dist < 125

def point_in_mouth(pt):
    num_vertices = len(mouth)
    x = pt[0]
    y = pt[1]
    inside = False

    p1 = mouth[0]
    for i in range(1, num_vertices + 1):
        p2 = mouth[i % num_vertices]
        if y > min(p1[1], p2[1]) and y <= max(p1[1], p2[1]) and x <= max(p1[0], p2[0]):
            x_intersection = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
            if p1[0] == p2[0] or x <= x_intersection:
                inside = not inside
        p1 = p2
    return inside

with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                    max_num_hands=2) as hands, mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as face_mesh:
    while webcam.isOpened():
        success, img = webcam.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        results_face = face_mesh.process(img_rgb)

        left_hand = []
        right_hand = []
        mouth = []

        h, w, ic = img.shape

        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                for i in mouth_landmarks:
                    lm = face_landmarks.landmark[i]
                    x, y = int(lm.x * w), int(lm.y * h)
                    mouth.append((x, y))
                for landmark in mouth:
                    cv2.circle(img, (landmark[0], landmark[1]), 3, (0, 255, 0), -1)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                for lm in hand_landmarks.landmark:
                    x_px = int(lm.x * w)
                    y_px = int(lm.y * h)
                    z_rel = lm.z
                    if label == 'Left':
                        left_hand.append((x_px, y_px, z_rel))
                    else:
                        right_hand.append((x_px, y_px, z_rel))

                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
                cv2.putText(img, label, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        current_time = time.time()

        if left_hand and right_hand and hands_touching():
            if hand_touch_start is None:
                hand_touch_start = current_time
            elif current_time - hand_touch_start >= 5:
                print('hands are touching')
                asyncio.run(sendNotif('Stop touching your hands'))
        else:
            hand_touch_start = None

        touching_mouth = False
        for fingertip in fingertip_landmarks:
            if left_hand and mouth and fingertip < len(left_hand):
                if point_in_mouth((left_hand[fingertip][0], left_hand[fingertip][1])):
                    touching_mouth = True
            if right_hand and mouth and fingertip < len(right_hand):
                if point_in_mouth((right_hand[fingertip][0], right_hand[fingertip][1])):
                    touching_mouth = True

        if touching_mouth:
            if face_touch_start is None:
                face_touch_start = current_time
            elif current_time - face_touch_start >= 5:
                print('mouth touching')
                asyncio.run(sendNotif('Stop touching your mouth'))
        else:
            face_touch_start = None

        cv2.imshow('Ishita', img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()
