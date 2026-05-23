import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)

    if mag_ab == 0 or mag_cb == 0:
        return 0

    cos_angle = dot / (mag_ab * mag_cb)
    cos_angle = max(-1, min(1, cos_angle))
    angle = math.degrees(math.acos(cos_angle))
    return int(angle)


cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read webcam.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            h, w, _ = frame.shape
            landmarks = results.pose_landmarks.landmark

            # Right arm
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            r_shoulder_point = (int(r_shoulder.x * w), int(r_shoulder.y * h))
            r_elbow_point = (int(r_elbow.x * w), int(r_elbow.y * h))
            r_wrist_point = (int(r_wrist.x * w), int(r_wrist.y * h))

            right_angle = calculate_angle(r_shoulder_point, r_elbow_point, r_wrist_point)

            # Left arm
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

            l_shoulder_point = (int(l_shoulder.x * w), int(l_shoulder.y * h))
            l_elbow_point = (int(l_elbow.x * w), int(l_elbow.y * h))
            l_wrist_point = (int(l_wrist.x * w), int(l_wrist.y * h))

            left_angle = calculate_angle(l_shoulder_point, l_elbow_point, l_wrist_point)

            # Show text
            cv2.putText(frame, f"Right elbow: {right_angle}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Left elbow: {left_angle}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Right arm dots
            cv2.circle(frame, r_shoulder_point, 8, (255, 0, 0), -1)
            cv2.circle(frame, r_elbow_point, 8, (0, 255, 0), -1)
            cv2.circle(frame, r_wrist_point, 8, (0, 0, 255), -1)

            # Left arm dots
            cv2.circle(frame, l_shoulder_point, 8, (255, 0, 255), -1)
            cv2.circle(frame, l_elbow_point, 8, (0, 255, 255), -1)
            cv2.circle(frame, l_wrist_point, 8, (255, 255, 0), -1)

        cv2.imshow("Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

# cd ~/mediapipe_robot
# source venv/bin/activate
# python3 test_pose.py
