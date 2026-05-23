import cv2
import numpy as np


def compute_zone_bounds(width: int) -> tuple[int, int]:
    """Return (x1, x2) pixel bounds of the middle third of a frame."""
    return width // 3, 2 * (width // 3)


def crop_to_zone(frame: np.ndarray, x1: int, x2: int) -> np.ndarray:
    """Return a copy of the middle vertical slice of frame."""
    return frame[:, x1:x2].copy()


def apply_gray_overlay(frame: np.ndarray, x1: int, x2: int, alpha: float = 0.55) -> None:
    """Darken the left and right thirds of frame in-place."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (x1, h), (40, 40, 40), -1)
    cv2.rectangle(overlay, (x2, 0), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_zone_lines(frame: np.ndarray, x1: int, x2: int) -> None:
    """Draw vertical boundary lines and a 'STAND HERE' label on frame in-place."""
    h = frame.shape[0]
    cv2.line(frame, (x1, 0), (x1, h), (255, 255, 255), 2)
    cv2.line(frame, (x2, 0), (x2, h), (255, 255, 255), 2)
    cv2.putText(
        frame,
        "STAND HERE",
        (x1 + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
