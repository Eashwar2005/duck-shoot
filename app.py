import cv2
import mediapipe as mp
import numpy as np

# ===============================
# Load & resize crosshair
# ===============================
crosshair_raw = cv2.imread("crosshair.png", cv2.IMREAD_UNCHANGED)
if crosshair_raw is None:
    raise FileNotFoundError("crosshair.png not found")

CROSSHAIR_SIZE = 35   # smaller, more accurate
crosshair = cv2.resize(crosshair_raw, (CROSSHAIR_SIZE, CROSSHAIR_SIZE))

# ===============================
# MediaPipe Hands
# ===============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ===============================
# Webcam
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# ===============================
# Fullscreen window
# ===============================
WINDOW_NAME = "Duck Shoot – Middle Finger Aim"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# ===============================
# Smoothing variables
# ===============================
smooth_x, smooth_y = None, None
SMOOTHING = 0.25   # lower = smoother, higher = faster

print("[INFO] Tracking middle finger MCP (landmark 9)")

# ===============================
# Main loop
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        # Draw landmarks (debug)
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # ===============================
        # Middle finger MCP (landmark 9)
        # ===============================
        lm = hand.landmark[9]

        raw_x = int(lm.x * w)
        raw_y = int(lm.y * h)

        # Smooth movement
        if smooth_x is None:
            smooth_x, smooth_y = raw_x, raw_y
        else:
            smooth_x = int(smooth_x * (1 - SMOOTHING) + raw_x * SMOOTHING)
            smooth_y = int(smooth_y * (1 - SMOOTHING) + raw_y * SMOOTHING)

        cx, cy = smooth_x, smooth_y

        # ===============================
        # Safe overlay
        # ===============================
        ch_h, ch_w = crosshair.shape[:2]

        x1 = max(0, cx - ch_w // 2)
        y1 = max(0, cy - ch_h // 2)
        x2 = min(w, x1 + ch_w)
        y2 = min(h, y1 + ch_h)

        ch_crop = crosshair[0:(y2 - y1), 0:(x2 - x1)]

        if ch_crop.shape[2] == 4:
            alpha = ch_crop[:, :, 3] / 255.0
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    alpha * ch_crop[:, :, c]
                    + (1 - alpha) * frame[y1:y2, x1:x2, c]
                )
        else:
            frame[y1:y2, x1:x2] = ch_crop

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()








