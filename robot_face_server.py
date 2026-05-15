from flask import Flask, Response
from pathlib import Path
import cv2
import numpy as np
import math
import time

app = Flask(__name__)

BASE = Path.home() / "mediapipe_robot"
FACE_STATE_FILE = BASE / "face_state.txt"

W, H = 640, 360

def read_face():
    try:
        return FACE_STATE_FILE.read_text().strip() or "neutral"
    except Exception:
        return "neutral"

def draw_face(face, t):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = (12, 12, 18)

    # background glow
    cv2.circle(frame, (W // 2, H // 2), 150, (35, 25, 70), -1)
    cv2.circle(frame, (W // 2, H // 2), 120, (45, 35, 95), -1)

    # head
    cx, cy = W // 2, H // 2 + 10
    cv2.ellipse(frame, (cx, cy), (130, 120), 0, 0, 360, (223, 199, 172), -1)
    cv2.ellipse(frame, (cx, cy), (130, 120), 0, 0, 360, (65, 55, 45), 4)

    # hair
    hair_pts = np.array([
        [cx - 115, cy - 95],
        [cx - 85, cy - 135],
        [cx - 35, cy - 85],
        [cx,      cy - 135],
        [cx + 45,  cy - 90],
        [cx + 95,  cy - 130],
        [cx + 125, cy - 70],
        [cx + 110, cy - 25],
        [cx - 110, cy - 25]
    ], np.int32)
    cv2.fillPoly(frame, [hair_pts], (18, 16, 28))

    blink = math.sin(t * 4.5) > 0.96
    bob = int(math.sin(t * 1.5) * 2)

    eye_y = cy - 35 + bob

    # default expression
    eye_open = 18
    mouth_type = "neutral"

    if face == "happy":
        eye_open = 16
        mouth_type = "smile"
    elif face == "surprised":
        eye_open = 28
        mouth_type = "o"
    elif face == "angry":
        eye_open = 12
        mouth_type = "flat"
    elif face == "wink":
        eye_open = 18
        mouth_type = "smile"
    elif face == "shy":
        eye_open = 14
        mouth_type = "small"
    elif face == "excited":
        eye_open = 22
        mouth_type = "bigsmile"

    # eyes
    if blink and face != "surprised":
        cv2.line(frame, (cx - 75, eye_y), (cx - 25, eye_y), (255, 255, 255), 4)
        cv2.line(frame, (cx + 25, eye_y), (cx + 75, eye_y), (255, 255, 255), 4)
    elif face == "wink":
        cv2.ellipse(frame, (cx - 50, eye_y), (35, eye_open), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, (cx + 50, eye_y), (35, eye_open), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(frame, (cx - 50, eye_y), 14, (0, 220, 255), -1)
        cv2.line(frame, (cx + 25, eye_y), (cx + 75, eye_y), (255, 255, 255), 4)
    else:
        cv2.ellipse(frame, (cx - 50, eye_y), (35, eye_open), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, (cx + 50, eye_y), (35, eye_open), 0, 0, 360, (255, 255, 255), -1)

        pupil_y = eye_y + (1 if face != "angry" else 0)
        cv2.circle(frame, (cx - 50, pupil_y), 14, (0, 220, 255), -1)
        cv2.circle(frame, (cx + 50, pupil_y), 14, (0, 220, 255), -1)
        cv2.circle(frame, (cx - 50, eye_y), 5, (255, 255, 255), -1)
        cv2.circle(frame, (cx + 50, eye_y), 5, (255, 255, 255), -1)

    # eyebrows
    if face == "angry":
        cv2.line(frame, (cx - 75, eye_y - 28), (cx - 25, eye_y - 20), (40, 40, 40), 5)
        cv2.line(frame, (cx + 25, eye_y - 20), (cx + 75, eye_y - 28), (40, 40, 40), 5)
    elif face == "surprised":
        cv2.line(frame, (cx - 75, eye_y - 22), (cx - 25, eye_y - 28), (40, 40, 40), 5)
        cv2.line(frame, (cx + 25, eye_y - 28), (cx + 75, eye_y - 22), (40, 40, 40), 5)
    else:
        cv2.line(frame, (cx - 75, eye_y - 30), (cx - 25, eye_y - 35), (40, 40, 40), 5)
        cv2.line(frame, (cx + 25, eye_y - 35), (cx + 75, eye_y - 30), (40, 40, 40), 5)

    # cheeks
    if face in ("happy", "shy", "excited"):
        cv2.circle(frame, (cx - 85, cy + 20), 18, (120, 80, 180), -1)
        cv2.circle(frame, (cx + 85, cy + 20), 18, (120, 80, 180), -1)

    # mouth
    if mouth_type == "smile":
        cv2.ellipse(frame, (cx, cy + 50), (45, 22), 0, 0, 180, (255, 255, 255), 5)
    elif mouth_type == "bigsmile":
        cv2.ellipse(frame, (cx, cy + 50), (60, 30), 0, 0, 180, (255, 255, 255), 5)
    elif mouth_type == "o":
        cv2.circle(frame, (cx, cy + 55), 18, (255, 255, 255), 5)
    elif mouth_type == "small":
        cv2.ellipse(frame, (cx, cy + 55), (25, 12), 0, 0, 180, (255, 255, 255), 4)
    elif mouth_type == "flat":
        cv2.line(frame, (cx - 35, cy + 55), (cx + 35, cy + 55), (255, 255, 255), 4)
    else:
        cv2.line(frame, (cx - 35, cy + 55), (cx + 35, cy + 55), (255, 255, 255), 4)

    cv2.putText(frame, face.upper(), (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 180), 3)

    return frame

def generate():
    while True:
        face = read_face()
        frame = draw_face(face, time.time())

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        jpg = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

        time.sleep(1 / 15)

@app.route("/")
def home():
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Robot Face</title>
      <style>
        html, body { margin:0; width:100%; height:100%; background:black; overflow:hidden; }
        img { width:100vw; height:100vh; object-fit:contain; display:block; }
      </style>
    </head>
    <body>
      <img src="/stream">
    </body>
    </html>
    """

@app.route("/stream")
def stream():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
