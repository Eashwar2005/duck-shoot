import cv2
import mediapipe as mp
import numpy as np
import random
import time

# ===============================
# Load & resize crosshair
# ===============================
crosshair_raw = cv2.imread("crosshair.png", cv2.IMREAD_UNCHANGED)
duck_raw = cv2.imread("duck.png", cv2.IMREAD_UNCHANGED)

if crosshair_raw is None:
    raise FileNotFoundError("crosshair.png not found")
if duck_raw is None:
    raise FileNotFoundError("duck.png not found")

CROSSHAIR_SIZE = 35
DUCK_W, DUCK_H = 80, 60

crosshair = cv2.resize(crosshair_raw, (CROSSHAIR_SIZE, CROSSHAIR_SIZE))
duck = cv2.resize(duck_raw, (DUCK_W, DUCK_H))

# ===============================
# MediaPipe Hands (YOUR VERSION)
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
# Smoothing variables (UNCHANGED)
# ===============================
smooth_x, smooth_y = None, None
SMOOTHING = 0.25

# ===============================
# Duck state
# ===============================
duck_x = -DUCK_W
duck_y = random.randint(120, 360)
duck_speed = 2

score = 0
last_fire = 0
FIRE_RATE = 0.15

print("[INFO] Duck Shoot started (good logic base)")

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

    firing = False
    cx, cy = None, None

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

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

        if smooth_x is None:
            smooth_x, smooth_y = raw_x, raw_y
        else:
            smooth_x = int(smooth_x * (1 - SMOOTHING) + raw_x * SMOOTHING)
            smooth_y = int(smooth_y * (1 - SMOOTHING) + raw_y * SMOOTHING)

        cx, cy = smooth_x, smooth_y

        # ===============================
        # Auto-fire (thumb bent + index extended)
        # ===============================
        thumb_bent = hand.landmark[4].y > hand.landmark[3].y
        index_extended = hand.landmark[8].y < hand.landmark[6].y

        if thumb_bent and index_extended:
            firing = True

    # ===============================
    # Move duck
    # ===============================
    duck_x += duck_speed
    if duck_x > w:
        duck_x = -DUCK_W
        duck_y = random.randint(120, 360)

    # ===============================
    # Hit detection
    # ===============================
    if firing and cx is not None:
        now = time.time()
        if now - last_fire > FIRE_RATE:
            last_fire = now
            print("AUTO FIRE")

            if duck_x < cx < duck_x + DUCK_W and duck_y < cy < duck_y + DUCK_H:
                score += 1
                print(f"HIT! Score = {score}")
                duck_x = -DUCK_W
                duck_y = random.randint(120, 360)

    # ===============================
    # Safe overlay function
    # ===============================
    def overlay_png(img, png, x, y):
        ph, pw = png.shape[:2]
        if x < 0 or y < 0 or x + pw > w or y + ph > h:
            return

        alpha = png[:, :, 3] / 255.0
        for c in range(3):
            img[y:y+ph, x:x+pw, c] = (
                alpha * png[:, :, c] +
                (1 - alpha) * img[y:y+ph, x:x+pw, c]
            )

    # Draw duck
    overlay_png(frame, duck, duck_x, duck_y)

    # Draw crosshair
    if cx is not None:
        overlay_png(
            frame,
            crosshair,
            cx - CROSSHAIR_SIZE // 2,
            cy - CROSSHAIR_SIZE // 2
        )

    # HUD
    cv2.putText(
        frame,
        f"Score: {score}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()












