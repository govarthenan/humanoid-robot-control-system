# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Real-time humanoid robot teleoperation via computer vision. A human operator faces a webcam; MediaPipe detects 33 body landmarks, joint angles are calculated geometrically, mapped through a calibration table to servo ranges, and streamed over TCP to a Raspberry Pi that drives the physical robot.

## Running the System

This project uses a raw venv (no pyproject.toml). Activate it before running anything:

```bash
source .venv/bin/activate
```

Or invoke directly:

```bash
.venv/bin/python3 full_body_robot_mapper.py    # Full-body control (production)
.venv/bin/python3 robot_mapper.py              # Arms-only, debug output (exit: q)
.venv/bin/python3 robot_shoulder_mapper.py     # Advanced 3D shoulder analysis (exit: q)
.venv/bin/python3 robot_face_server.py         # Flask face visualization at :8080
.venv/bin/python3 test_pose.py                 # Pose detection smoke test
.venv/bin/python3 robot_sender_test.py         # Socket connectivity test
```

Exit `full_body_robot_mapper.py` with the ESC key.

## Dependencies

No lock file or package manifest exists. Install manually:

```bash
pip install opencv-python mediapipe numpy flask
```

Note: `flask` is not currently installed in `.venv` — `robot_face_server.py` will fail without it.

## Architecture & Data Flow

```
Webcam → MediaPipe (33 landmarks) → Angle Calculation → Calibration Mapping → Smoothing → TCP Socket → Raspberry Pi
                                                                                         └→ face_state.txt → Flask Server
```

**Script responsibilities:**

| File | Role |
|---|---|
| `full_body_robot_mapper.py` | Production: head + arms + legs + face expression |
| `robot_mapper.py` | Dev: 2D elbow/shoulder angles only |
| `robot_shoulder_mapper.py` | Dev: 3D shoulder kinematics (abduction, flexion, rotation) |
| `robot_face_server.py` | Flask server rendering animated SVG face at `:8080` |

## Key Configuration (hardcoded in `full_body_robot_mapper.py`)

- **Robot IP/Port:** `PI_IP = "100.99.129.111"`, `PORT = 5005` — change to target your Pi.
- **CALIBRATION dict:** Maps each joint's human angle range → servo range. Adjust when robot is rebuilt or joints change.
- **Face state file:** `~/mediapipe_robot/face_state.txt` — inter-process IPC between mapper and face server.

## Core Patterns

**Calibration mapping** (`map_range`): linear interpolation from human angle domain to servo angle domain. Every joint has `human_min`, `human_max`, `robot_min`, `robot_max`. Always clamp output before sending.

**Smoothing** (`smooth` function): exponential moving average with per-joint alpha. Lower alpha = more stable but laggier. `R_ROT` uses a hardcoded split (0.6/0.4) for extra stability — this is intentional.

**TCP command format:** `"HEAD_YAW=95,L_ELBOW=140,...,FACE=happy\n"` — newline-terminated, one frame per send.

**Facial expression detection:** heuristic based on hand landmark positions relative to face/shoulder landmarks. Not ML-based.

## Tests

No pytest setup. Two manual integration tests exist:

```bash
.venv/bin/python3 test_pose.py           # Validates MediaPipe + angle math (no Pi needed)
.venv/bin/python3 robot_sender_test.py   # Sends hardcoded angles to Pi (Pi must be reachable)
```
