import cv2
import mediapipe as mp
import math
from desktop_notifier import DesktopNotifier
import asyncio

notifier = DesktopNotifier()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)

async def sendNotif():
    n = await notifier.send(title='HANDS APART!', message='Stop touching your hands')

def calculate_dist(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def hands_touching():
    left_index_finger = left_hand[8]
    right_index_finger = right_hand[8]
    dist = calculate_dist(left_index_finger, right_index_finger)
    return dist < 125

with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                    max_num_hands=2) as hands:
    while webcam.isOpened():
        success, img = webcam.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        left_hand = []
        right_hand = []

        if results.multi_hand_landmarks and results.multi_handedness:
            h, w, _ = img.shape

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

        if left_hand and right_hand and hands_touching():
            asyncio.run(sendNotif())
            print('hands are touching')
        else:
            print('hand not touching')

        cv2.imshow('Ishita', img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()
