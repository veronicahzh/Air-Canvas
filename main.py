import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# gesture helpers

def finger_is_up(hand_landmarks, tip_id, pip_id):
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

def is_fist(hand_landmarks):
    idx = finger_is_up(hand_landmarks, 8, 6)
    mid = finger_is_up(hand_landmarks, 12, 10)
    ring = finger_is_up(hand_landmarks, 16, 14)
    pink = finger_is_up(hand_landmarks, 20, 18)
    return not (idx or mid or ring or pink)

def is_open_palm(hand_landmarks):
    idx = finger_is_up(hand_landmarks, 8, 6)
    mid = finger_is_up(hand_landmarks, 12, 10)
    ring = finger_is_up(hand_landmarks, 16, 14)
    pink = finger_is_up(hand_landmarks, 20, 18)
    return idx and mid and ring and pink

def is_draw_gesture(hand_landmarks):
    # Draw when only index finger is up
    idx = finger_is_up(hand_landmarks, 8, 6)
    mid = finger_is_up(hand_landmarks, 12, 10)
    ring = finger_is_up(hand_landmarks, 16, 14)
    pink = finger_is_up(hand_landmarks, 20, 18)
    return idx and not (mid or ring or pink)
