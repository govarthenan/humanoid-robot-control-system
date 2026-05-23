import cv2
import mediapipe as mp
import math

# ==========================================
# HUMAN TO ROBOT ANGLE MAPPER
# ==========================================
# This program:
# 1. reads webcam video
# 2. detects body landmarks using MediaPipe
# 3. calculates arm angles
# 4. converts them into robot servo-style values
# 5. shows everything on screen
#
# For now:
# - runs only on your Mac
# - does NOT control the robot yet
# ==========================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    """
    Calculate angle ABC in degrees using 2D points.
    a, b, c are (x, y) points.
    b is the joint point.
    """
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


def map_range(value, in_min, in_max, out_min, out_max):
    """
    Map a value from one range to another.
    """
    if in_max == in_min:
        return out_min

    mapped = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return int(mapped)


def clamp(value, min_value, max_value):
    """
    Keep value inside safe range.
    """
    return max(min_value, min(value, max_value))


def human_to_robot_elbow(human_angle):
    """
    Convert human elbow angle to robot servo angle.
    You will adjust these numbers later after real robot testing.
    """
    robot_angle = map_range(human_angle, 30, 170, 20, 140)
    return clamp(robot_angle, 20, 140)


def human_to_robot_shoulder(human_angle):
    """
    Convert human shoulder angle to robot servo angle.
    These are temporary estimated values for software testing.
    """
    robot_angle = map_range(human_angle, 20, 160, 40, 150)
    return clamp(robot_angle, 40, 150)


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

        # Flip like mirror so it feels natural
        frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            h, w, _ = frame.shape
            lm = results.pose_landmarks.landmark

            # -----------------------------
            # LEFT ARM POINTS
            # -----------------------------
            l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            l_elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]

            l_shoulder_pt = (int(l_shoulder.x * w), int(l_shoulder.y * h))
            l_elbow_pt = (int(l_elbow.x * w), int(l_elbow.y * h))
            l_wrist_pt = (int(l_wrist.x * w), int(l_wrist.y * h))
            l_hip_pt = (int(l_hip.x * w), int(l_hip.y * h))

            # -----------------------------
            # RIGHT ARM POINTS
            # -----------------------------
            r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            r_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

            r_shoulder_pt = (int(r_shoulder.x * w), int(r_shoulder.y * h))
            r_elbow_pt = (int(r_elbow.x * w), int(r_elbow.y * h))
            r_wrist_pt = (int(r_wrist.x * w), int(r_wrist.y * h))
            r_hip_pt = (int(r_hip.x * w), int(r_hip.y * h))

            # -----------------------------
            # HUMAN ANGLES
            # -----------------------------
            left_elbow_human = calculate_angle(l_shoulder_pt, l_elbow_pt, l_wrist_pt)
            right_elbow_human = calculate_angle(r_shoulder_pt, r_elbow_pt, r_wrist_pt)

            left_shoulder_human = calculate_angle(l_hip_pt, l_shoulder_pt, l_elbow_pt)
            right_shoulder_human = calculate_angle(r_hip_pt, r_shoulder_pt, r_elbow_pt)

            # -----------------------------
            # ROBOT COMMAND ANGLES
            # -----------------------------
            left_elbow_robot = human_to_robot_elbow(left_elbow_human)
            right_elbow_robot = human_to_robot_elbow(right_elbow_human)

            left_shoulder_robot = human_to_robot_shoulder(left_shoulder_human)
            right_shoulder_robot = human_to_robot_shoulder(right_shoulder_human)

            # -----------------------------
            # DRAW JOINT DOTS
            # -----------------------------
            cv2.circle(frame, l_shoulder_pt, 6, (255, 0, 255), -1)
            cv2.circle(frame, l_elbow_pt, 6, (0, 255, 255), -1)
            cv2.circle(frame, l_wrist_pt, 6, (255, 255, 0), -1)

            cv2.circle(frame, r_shoulder_pt, 6, (255, 0, 0), -1)
            cv2.circle(frame, r_elbow_pt, 6, (0, 255, 0), -1)
            cv2.circle(frame, r_wrist_pt, 6, (0, 0, 255), -1)

            # -----------------------------
            # SHOW HUMAN ANGLES
            # -----------------------------
            cv2.putText(
                frame, f"L Elbow Human: {left_elbow_human}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
            cv2.putText(
                frame, f"R Elbow Human: {right_elbow_human}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"L Shoulder Human: {left_shoulder_human}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"R Shoulder Human: {right_shoulder_human}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            # -----------------------------
            # SHOW ROBOT ANGLES
            # -----------------------------
            cv2.putText(
                frame, f"L_ELBOW_SERVO: {left_elbow_robot}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"R_ELBOW_SERVO: {right_elbow_robot}", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"L_SHOULDER_SERVO: {left_shoulder_robot}",
                (20, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"R_SHOULDER_SERVO: {right_shoulder_robot}",
                (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            # Also print robot values in Terminal for later use
            print(
                f"L_ELBOW={left_elbow_robot}, "
                f"R_ELBOW={right_elbow_robot}, "
                f"L_SHOULDER={left_shoulder_robot}, "
                f"R_SHOULDER={right_shoulder_robot}"
            )

        cv2.imshow("Human to Robot Angle Mapper", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
