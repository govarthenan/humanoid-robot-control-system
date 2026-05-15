import cv2
import mediapipe as mp
import math
import socket
from pathlib import Path

FACE_STATE_FILE = Path.home() / "mediapipe_robot" / "face_state.txt"

def save_face(face_name):
    try:
        FACE_STATE_FILE.write_text(face_name)
    except Exception:
        pass
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# =========================
# NETWORK
# =========================
PI_IP = "100.99.129.111"
PORT = 5005

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((PI_IP, PORT))
print("Connected to Raspberry Pi")

# =========================
# CALIBRATION (UNCHANGED)
# =========================
CALIBRATION = {
    "HEAD_YAW": {"human_min": -20, "human_max": 20, "robot_min": 50, "robot_max": 140},
    "HEAD_PITCH": {"human_min": -20, "human_max": 20, "robot_min": 130, "robot_max": 70},

    "L_ELBOW": {"human_min": 75, "human_max": 145, "robot_min": 135, "robot_max": 0},
    "L_ABD": {"human_min": 10, "human_max": 90, "robot_min": 25, "robot_max": 140},
    "L_FLEX": {"human_min": -140, "human_max": 90, "robot_min": 50, "robot_max": 170},
    "L_ROT": {"human_min": -20, "human_max": 90, "robot_min": 0, "robot_max": 180},

    "R_ELBOW": {"human_min": 75, "human_max": 145, "robot_min": 20, "robot_max": 180},
    "R_ABD": {"human_min": 10, "human_max": 100, "robot_min": 150, "robot_max": 40},
    "R_FLEX": {"human_min": -140, "human_max": 90, "robot_min": 180, "robot_max": 60},
    "R_ROT": {"human_min": 10, "human_max": 110, "robot_min": 180, "robot_max": 40},

    "L_HIP": {"human_min": -70, "human_max": 70, "robot_min": 80, "robot_max": 150},
    "L_KNEE": {"human_min": 45, "human_max": 170, "robot_min": 100, "robot_max": 0},
    "L_ANKLE": {"human_min": 70, "human_max": 155, "robot_min": 100, "robot_max": 180},

    "R_HIP": {"human_min": -70, "human_max": 70, "robot_min": 30, "robot_max": 100},
    "R_KNEE": {"human_min": 45, "human_max": 170, "robot_min": 80, "robot_max": 180},
    "R_ANKLE": {"human_min": 70, "human_max": 155, "robot_min": 0, "robot_max": 60},
}

# =========================
# HELPERS
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def map_range(v, a, b, c, d):
    v = clamp(v, a, b)
    return int((v - a) * (d - c) / (b - a) + c)

def calibrate(name, val):
    c = CALIBRATION[name]
    return map_range(val, c["human_min"], c["human_max"], c["robot_min"], c["robot_max"])

def smooth(name, val, alpha=0.3):
    if name not in smooth.prev:
        smooth.prev[name] = val
    smooth.prev[name] = int(alpha * val + (1 - alpha) * smooth.prev[name])
    return smooth.prev[name]
smooth.prev = {}

def draw_mark(frame, landmark, idx, color, label):
    h, w = frame.shape[:2]
    x = int(landmark[idx].x * w)
    y = int(landmark[idx].y * h)
    cv2.circle(frame, (x, y), 6, color, -1)
    cv2.putText(
        frame,
        label,
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )

# =========================
# ANGLE
# =========================
def angle(a, b, c):
    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag = math.hypot(*ab) * math.hypot(*cb)
    if mag == 0:
        return 0
    return math.degrees(math.acos(clamp(dot / mag, -1, 1)))
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def angle_by_distance(a, b, c):
    ab = dist(a, b)
    bc = dist(b, c)
    ac = dist(a, c)

    if ab == 0 or bc == 0:
        return 0

    cos_val = (ab * ab + bc * bc - ac * ac) / (2 * ab * bc)
    cos_val = clamp(cos_val, -1, 1)
    return math.degrees(math.acos(cos_val))
# =========================
# ROTATION BASE
# =========================
def arm_rot(shoulder, elbow, wrist):
    dx = wrist.x - elbow.x
    dy = wrist.y - elbow.y

    angle_val = math.degrees(math.atan2(dy, dx))

    if angle_val > 180:
        angle_val -= 360
    if angle_val < -180:
        angle_val += 360

    return clamp(angle_val / 2, -90, 90)

# Stronger right ABD signal
def right_abd_signal(shoulder, elbow, wrist):
    # Blend elbow + wrist relative to shoulder so movement is more visible.
    vx = (elbow.x - shoulder.x) * 0.7 + (wrist.x - shoulder.x) * 0.3
    vy = (elbow.y - shoulder.y) * 0.7 + (wrist.y - shoulder.y) * 0.3

    # 0 when arm is hanging down, bigger when arm moves sideways.
    raw = math.degrees(math.atan2(vx, vy))
    return clamp(raw, -90, 90)
def get_face_expression(lm):
    nose = lm[0]

    ls = lm[11]
    rs = lm[12]

    lw = lm[15]
    rw = lm[16]

    # hand positions
    left_up = lw.y < ls.y - 0.05
    right_up = rw.y < rs.y - 0.05

    both_up = lw.y < nose.y and rw.y < nose.y

    hands_close = abs(lw.x - rw.x) < 0.18
    hands_wide = abs(lw.x - rw.x) > 0.5

    if both_up:
        return "surprised"

    if left_up and right_up:
        return "excited"

    if right_up:
        return "happy"

    if left_up:
        return "wink"

    if hands_wide:
        return "angry"

    if hands_close:
        return "shy"

    return "neutral"
# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (960, 720))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = pose.process(rgb)

        if not res.pose_landmarks:
            continue

        lm = res.pose_landmarks.landmark
        final = {}

        # facial expression
        final["FACE"] = get_face_expression(lm)
        save_face(final["FACE"])
        # Draw skeleton + key dots

        mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        draw_mark(frame, lm, 0,  (0, 255, 255), "N")
        draw_mark(frame, lm, 11, (255, 0, 0), "LS")
        draw_mark(frame, lm, 13, (255, 0, 0), "LE")
        draw_mark(frame, lm, 15, (255, 0, 0), "LW")
        draw_mark(frame, lm, 12, (0, 0, 255), "RS")
        draw_mark(frame, lm, 14, (0, 0, 255), "RE")
        draw_mark(frame, lm, 16, (0, 0, 255), "RW")
        draw_mark(frame, lm, 23, (0, 200, 200), "LH")
        draw_mark(frame, lm, 24, (200, 200, 0), "RH")
        draw_mark(frame, lm, 23, (0, 200, 200), "LH")
        draw_mark(frame, lm, 25, (0, 200, 200), "LK")
        draw_mark(frame, lm, 27, (0, 200, 200), "LA")
        draw_mark(frame, lm, 31, (0, 200, 200), "LF")

        draw_mark(frame, lm, 24, (200, 200, 0), "RH")
        draw_mark(frame, lm, 26, (200, 200, 0), "RK")
        draw_mark(frame, lm, 28, (200, 200, 0), "RA")
        draw_mark(frame, lm, 32, (200, 200, 0), "RF")

        # ===== HEAD =====
        nose = lm[0]
        le = lm[7]
        re = lm[8]

        yaw = (nose.x - (le.x + re.x) / 2) * 400
        pitch = (nose.y - (le.y + re.y) / 2) * 400

        final["HEAD_YAW"] = calibrate("HEAD_YAW", yaw)
        final["HEAD_PITCH"] = calibrate("HEAD_PITCH", pitch)

        # ===== LEFT =====
        ls, lelb, lw = lm[11], lm[13], lm[15]

        final["L_ELBOW"] = calibrate("L_ELBOW", angle(ls, lelb, lw))
        final["L_FLEX"] = calibrate("L_FLEX", (lw.y - ls.y) * 200)
        # ===== LEFT ABD FIX =====
        l_abd_val = angle(lelb, ls, lm[23])   # elbow-shoulder-hip
        final["L_ABD"] = calibrate("L_ABD", l_abd_val)

        l_rot_val = arm_rot(ls, lelb, lw)
        if abs(l_rot_val) < 10:
            l_rot_val = (lw.x - ls.x) * 200

        final["L_ROT"] = calibrate("L_ROT", l_rot_val)
        final["L_ROT"] = smooth("L_ROT", final["L_ROT"], alpha=0.2)

        # ===== RIGHT =====
        rs, relb, rw = lm[12], lm[14], lm[16]

        final["R_ELBOW"] = calibrate("R_ELBOW", angle(rs, relb, rw))
        final["R_FLEX"] = calibrate("R_FLEX", (rw.y - rs.y) * 200)

        # FIXED R_ABD (angle-based)
        r_abd_val = angle(relb, rs, lm[24])
        final["R_ABD"] = calibrate("R_ABD", r_abd_val)

        # ===== RIGHT ROTATION FINAL REAL FIX =====

        # vector from wrist to elbow
        vx = rw.x - relb.x
        vy = rw.y - relb.y

        # angle of forearm
        r_rot_val = math.degrees(math.atan2(vy, vx))

        # normalize
        if r_rot_val > 180:
           r_rot_val -= 360
        if r_rot_val < -180:
           r_rot_val += 360

        #  IMPORTANT: amplify strongly
        r_rot_val *= 3.5

        # clamp to safe range
        r_rot_val = clamp(r_rot_val, -60, 60)

        # stabilize
        if "R_ROT" not in smooth.prev:
         smooth.prev["R_ROT"] = r_rot_val

         smooth.prev["R_ROT"] = 0.6 * smooth.prev["R_ROT"] + 0.4 * r_rot_val

        # map to servo
        final["R_ROT"] = calibrate("R_ROT", smooth.prev["R_ROT"])
        final["R_ROT"] = smooth("R_ROT", final["R_ROT"], alpha=0.2)
        # ===== LEFT LEG =====
        l_hip = lm[23]
        l_knee = lm[25]
        l_ankle = lm[27]
        l_foot = lm[31]

        l_hip_val = (l_knee.y - l_hip.y) * 300

        final["L_HIP"] = calibrate(
            "L_HIP",
            l_hip_val
        )

        final["L_KNEE"] = calibrate(
            "L_KNEE",
            angle_by_distance(l_hip, l_knee, l_ankle)
        )

        final["L_ANKLE"] = calibrate(
            "L_ANKLE",
            angle_by_distance(l_knee, l_ankle, l_foot)
        )

        # ===== RIGHT LEG =====
        r_hip = lm[24]
        r_knee = lm[26]
        r_ankle = lm[28]
        r_foot = lm[32]

        r_hip_val = (r_knee.y - r_hip.y) * 300

        final["R_HIP"] = calibrate(
            "R_HIP",
            r_hip_val
        )

        final["R_KNEE"] = calibrate(
            "R_KNEE",
            angle_by_distance(r_hip, r_knee, r_ankle)
        )

        final["R_ANKLE"] = calibrate(
            "R_ANKLE",
            angle_by_distance(r_knee, r_ankle, r_foot)
        )

        # ===== SMOOTH (others) =====
        for k in final:

            # do not smooth FACE text
            if k == "FACE":
                continue

            # keep custom smoothing disabled for these
            if k not in ["R_ROT", "L_ROT", "R_ABD"]:
                final[k] = smooth(k, final[k])

        # ===== SEND =====
        msg_parts = []

        for k, v in final.items():
            msg_parts.append(f"{k}={v}")

        msg = ",".join(msg_parts)
        print("SENDING:", msg)
        client.sendall((msg + "\n").encode())

        # ===== DISPLAY =====
        y = 30
        for k, v in final.items():
            cv2.putText(
                frame,
                f"{k}:{v}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y += 25

        cv2.imshow("Robot", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
client.close()