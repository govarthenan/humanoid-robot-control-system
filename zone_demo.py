"""Visual smoke-test for the zone filter — no Pi connection needed.

Shows the three-slice view with MediaPipe pose running on the middle crop.
Press ESC to exit.
"""

import cv2
import mediapipe as mp
from zone_filter import apply_gray_overlay, compute_zone_bounds, crop_to_zone, draw_zone_lines

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (960, 720))
        h, w = frame.shape[:2]
        x1, x2 = compute_zone_bounds(w)

        middle = crop_to_zone(frame, x1, x2)
        rgb = cv2.cvtColor(middle, cv2.COLOR_BGR2RGB)

        res = pose.process(rgb)

        if res.pose_landmarks:
            mp_drawing.draw_landmarks(middle, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frame[:, x1:x2] = middle

        apply_gray_overlay(frame, x1, x2)
        draw_zone_lines(frame, x1, x2)

        status = "TRACKING" if res.pose_landmarks else "NO PERSON IN ZONE"
        color = (0, 255, 0) if res.pose_landmarks else (0, 200, 255)
        cv2.putText(frame, status, (x1 + 10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Zone Filter Demo", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
