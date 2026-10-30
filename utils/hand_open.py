"""MCAT gripper decisions, thresholds and state transitions.
Copyright (c) 2025 BIT807s.
"""
import math
import numpy as np

class HandStateManager:
    'Stabilize gripper transitions using consecutive hand-state decisions.'

    def __init__(self, confidence_threshold=0.8, frame_threshold=3, hand='l', close=0.02, open=0.06):
        'Initialize the state counters and open/closed command values.\n\nThe confidence threshold is stored but is not used as a decision gate.'
        self.confidence_threshold = confidence_threshold
        self.frame_threshold = frame_threshold
        self.is_hand_open = False
        self.gripper_value = 0.0
        self.open_counter = 0
        self.closed_counter = 0
        self.open_gripper_value = open
        self.closed_gripper_value = close
        self.switch = False
        self.hand = hand

    def update(self, hand_landmarks):
        'Update the gripper command from hand landmarks.\n\nMissing observations reset the counters and hold the current command.'
        if hand_landmarks is None:
            self.open_counter = 0
            self.closed_counter = 0
            return self.gripper_value
        if self.hand == 'l':
            is_gripper, avg_ratio, confidence_score = is_hand_gripping(hand_landmarks, th=43)
        if self.hand == 'r':
            is_gripper, avg_ratio, confidence_score = is_hand_gripping(hand_landmarks, th=48)
        if not is_gripper:
            self.open_counter += 1
            self.closed_counter = 0
        else:
            self.closed_counter += 1
            self.open_counter = 0
        self.switch = False
        if self.open_counter >= self.frame_threshold and (not is_gripper):
            self.switch = True
            self.is_hand_open = True
            self.gripper_value = self.open_gripper_value
            print(self.hand, f'Hand open------  ', avg_ratio)
        elif self.closed_counter >= self.frame_threshold and is_gripper:
            self.switch = True
            self.is_hand_open = False
            self.gripper_value = self.closed_gripper_value
            print(self.hand, f'------Hand closed', avg_ratio)
        return self.gripper_value

    def get_status_text(self):
        'Return an English status label and counter-based stability indicator.'
        if self.is_hand_open:
            status = '---------Open-----------'
            confidence = self.open_counter / self.frame_threshold
        else:
            status = '---------------------Closed'
            confidence = self.closed_counter / self.frame_threshold
        confidence = min(confidence, 1.0)
        return f'Hand state: {status} (Stability: {confidence:.2f})'

def is_hand_gripping(hand_landmarks, th) -> bool:
    landmarks = hand_landmarks
    finger_L2 = []
    l_finger = [7, 11, 15]
    for idx in l_finger:
        distance = math.sqrt((landmarks[idx].x - landmarks[3].x) ** 2 + (landmarks[idx].y - landmarks[3].y) ** 2 + (landmarks[idx].z - landmarks[3].z) ** 2) * 1000
        finger_L2.append(distance)
    finger_L2_mean = np.array(finger_L2).mean()
    if finger_L2_mean < th:
        gripper = True
    else:
        gripper = False
    return (gripper, finger_L2_mean, finger_L2_mean)
