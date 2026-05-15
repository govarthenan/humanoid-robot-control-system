import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def vec(a, b):
    return [b[0] - a[0], b[1] - a[1], b[2] - a[2]]


def dot(u, v):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]


def norm(v):
    return math.sqrt(dot(v, v))


def unit(v):
    n = norm(v)
    if n == 0:
        return [0.0, 0.0, 0.0]
    return [v[0]/n, v[1]/n, v[2]/n]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def angle_between(u, v):
    nu = norm(u)
    nv = norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    c = clamp(dot(u, v) / (nu * nv), -1.0, 1.0)
    return math.degrees(math.acos(c))


def project_onto_plane(v, plane_normal):
    n = unit(plane_normal)
    d = dot(v, n)
    return [v[0] - d*n[0], v[1] - d*n[1], v[2] - d*n[2]]


def signed_angle_on_plane(v1, v2, plane_normal):
    a = unit(project_onto_plane(v1, plane_normal))
    b = unit(project_onto_plane(v2, plane_normal))
    if norm(a) == 0 or norm(b) == 0:
        return 0.0

    cross = [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]
    s = dot(cross, unit(plane_normal))
    c = clamp(dot(a, b), -1.0, 1.0)
    return math.degrees(math.atan2(s, c))


def map_range(value, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return out_min
    value = clamp(value, in_min, in_max)
    return int((value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


def get_xyz(world_landmarks, idx):
    lm = world_landmarks[idx]
    return [lm.x, lm.y, lm.z]


def shoulder_metrics(side_name, shoulder, elbow, wrist, hip, opposite_shoulder):
    # Body reference axes
    torso_down = vec(shoulder, hip)                # shoulder -> hip
    shoulder_line = vec(shoulder, opposite_shoulder)  # shoulder -> opposite shoulder
    upper_arm = vec(shoulder, elbow)               # shoulder -> elbow
    forearm = vec(elbow, wrist)                    # elbow -> wrist

    # Approx body forward axis from torso and shoulder line
    # right-hand-like body frame proxy
    body_forward = [
        shoulder_line[1]*torso_down[2] - shoulder_line[2]*torso_down[1],
        shoulder_line[2]*torso_down[0] - shoulder_line[0]*torso_down[2],
        shoulder_line[0]*torso_down[1] - shoulder_line[1]*torso_down[0]
    ]

    # 1) Elbow flexion/extension
    elbow_flex = angle_between(
        [shoulder[0]-elbow[0], shoulder[1]-elbow[1], shoulder[2]-elbow[2]],
        [wrist[0]-elbow[0], wrist[1]-elbow[1], wrist[2]-elbow[2]]
    )

    # 2) Shoulder abduction/adduction
    # angle on frontal plane: plane normal ~ body forward
    shoulder_abd = abs(signed_angle_on_plane(torso_down, upper_arm, body_forward))

    # 3) Shoulder flexion/extension
    # angle on sagittal plane: plane normal ~ shoulder line
    shoulder_flex = signed_angle_on_plane(torso_down, upper_arm, shoulder_line)

    # 4) Shoulder rotation proxy
    # uses forearm orientation around upper arm axis
    # NOT true anatomical internal/external rotation, just a robot-friendly estimate
    forearm_proj = project_onto_plane(forearm, upper_arm)
    torso_proj = project_onto_plane(torso_down, upper_arm)
    shoulder_rot_proxy = signed_angle_on_plane(torso_proj, forearm_proj, upper_arm)

    # Robot-style mapped values (temporary ranges)
    robot = {
        "elbow_servo": map_range(elbow_flex, 20, 170, 20, 140),
        "shoulder_abd_servo": map_range(shoulder_abd, 0, 120, 40, 150),
        "shoulder_flex_servo": map_range(shoulder_flex, -90, 90, 40, 150),
        "shoulder_rot_servo": map_range(shoulder_rot_proxy, -90, 90, 50, 130),
    }

    human = {
        "elbow_flex": int(elbow_flex),
        "shoulder_abd": int(shoulder_abd),
        "shoulder_flex": int(shoulder_flex),
        "shoulder_rot_proxy": int(shoulder_rot_proxy),
    }

    return human, robot


cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Failed to read webcam.")
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        if results.pose_world_landmarks and results.pose_landmarks:
            wlm = results.pose_world_landmarks.landmark
            ilm = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Left side world points
            l_sh = get_xyz(wlm, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            l_el = get_xyz(wlm, mp_pose.PoseLandmark.LEFT_ELBOW.value)
            l_wr = get_xyz(wlm, mp_pose.PoseLandmark.LEFT_WRIST.value)
            l_hi = get_xyz(wlm, mp_pose.PoseLandmark.LEFT_HIP.value)

            # Right side world points
            r_sh = get_xyz(wlm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            r_el = get_xyz(wlm, mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            r_wr = get_xyz(wlm, mp_pose.PoseLandmark.RIGHT_WRIST.value)
            r_hi = get_xyz(wlm, mp_pose.PoseLandmark.RIGHT_HIP.value)

            left_human, left_robot = shoulder_metrics("L", l_sh, l_el, l_wr, l_hi, r_sh)
            right_human, right_robot = shoulder_metrics("R", r_sh, r_el, r_wr, r_hi, l_sh)

            # Draw some 2D markers for clarity
            for idx in [
                mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                mp_pose.PoseLandmark.LEFT_ELBOW.value,
                mp_pose.PoseLandmark.LEFT_WRIST.value,
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                mp_pose.PoseLandmark.RIGHT_WRIST.value
            ]:
                p = ilm[idx]
                x, y = int(p.x * w), int(p.y * h)
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)

            y = 30
            lines = [
                f"L elbow flex: {left_human['elbow_flex']}",
                f"L shoulder abd/add: {left_human['shoulder_abd']}",
                f"L shoulder flex/ext: {left_human['shoulder_flex']}",
                f"L shoulder rot proxy: {left_human['shoulder_rot_proxy']}",
                f"R elbow flex: {right_human['elbow_flex']}",
                f"R shoulder abd/add: {right_human['shoulder_abd']}",
                f"R shoulder flex/ext: {right_human['shoulder_flex']}",
                f"R shoulder rot proxy: {right_human['shoulder_rot_proxy']}",
                "--- ROBOT COMMANDS ---",
                f"L_ELBOW={left_robot['elbow_servo']}",
                f"L_SHOULDER_ABD={left_robot['shoulder_abd_servo']}",
                f"L_SHOULDER_FLEX={left_robot['shoulder_flex_servo']}",
                f"L_SHOULDER_ROT={left_robot['shoulder_rot_servo']}",
                f"R_ELBOW={right_robot['elbow_servo']}",
                f"R_SHOULDER_ABD={right_robot['shoulder_abd_servo']}",
                f"R_SHOULDER_FLEX={right_robot['shoulder_flex_servo']}",
                f"R_SHOULDER_ROT={right_robot['shoulder_rot_servo']}",
            ]

            for line in lines:
                cv2.putText(
                    frame, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                )
                y += 24

            print(
                f"L_ABD={left_robot['shoulder_abd_servo']} "
                f"L_FLEX={left_robot['shoulder_flex_servo']} "
                f"L_ROT={left_robot['shoulder_rot_servo']} "
                f"L_ELBOW={left_robot['elbow_servo']} | "
                f"R_ABD={right_robot['shoulder_abd_servo']} "
                f"R_FLEX={right_robot['shoulder_flex_servo']} "
                f"R_ROT={right_robot['shoulder_rot_servo']} "
                f"R_ELBOW={right_robot['elbow_servo']}"
            )

        cv2.imshow("Robot Shoulder Mapper", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()