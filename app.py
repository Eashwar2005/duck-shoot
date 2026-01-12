import cv2
import mediapipe as mp
import numpy as np
import random
import time

# ===============================
# Load assets
# ===============================
bg = cv2.imread("background.png" )
crosshair_raw = cv2.imread("crosshair.png", cv2.IMREAD_UNCHANGED)
duck_raw = cv2.imread("duck.png", cv2.IMREAD_UNCHANGED)

if bg is None or crosshair_raw is None or duck_raw is None:
    raise FileNotFoundError("Required image missing")

BG_W, BG_H = 1280, 720
bg = cv2.resize(bg, (BG_W, BG_H))

# ===============================
# Sizes (UPDATED)
# ===============================
CROSSHAIR_SIZE = 35
DUCK_W, DUCK_H = 110, 80   # bigger duck

crosshair = cv2.resize(crosshair_raw, (CROSSHAIR_SIZE, CROSSHAIR_SIZE))
duck = cv2.resize(duck_raw, (DUCK_W, DUCK_H))

# ===============================
# Darken crosshair
# ===============================
crosshair[:, :, :3] = (crosshair[:, :, :3] * 0.55).astype(np.uint8)

# ===============================
# MediaPipe
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ===============================
# Camera
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# ===============================
# Fullscreen window
# ===============================
WINDOW_NAME = "Duck Shoot"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ===============================
# Smoothing (UNCHANGED)
# ===============================
smooth_x, smooth_y = None, None
SMOOTHING = 0.25

# ===============================
# Ducks (MULTI)
# ===============================
BASE_SPEED = 2.6  # +30%
ducks = []

def spawn_duck():
    edge = random.choice(["left", "right", "top", "bottom"])
    speed = random.uniform(2.0, 3.5)  # base + randomness

    if edge == "left":
        x = -DUCK_W
        y = random.randint(0, BG_H - DUCK_H)
        angle = random.uniform(-0.3, 0.3)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)

    elif edge == "right":
        x = BG_W
        y = random.randint(0, BG_H - DUCK_H)
        angle = random.uniform(np.pi - 0.3, np.pi + 0.3)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)

    elif edge == "top":
        x = random.randint(0, BG_W - DUCK_W)
        y = -DUCK_H
        angle = random.uniform(0.8, 2.3)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)

    else:  # bottom
        x = random.randint(0, BG_W - DUCK_W)
        y = BG_H
        angle = random.uniform(-2.3, -0.8)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)

    return {
        "x": float(x),
        "y": float(y),
        "vx": vx,
        "vy": vy
    }


for _ in range(2):  # initial ducks = 2
    ducks.append(spawn_duck())

# ===============================
# Game state
# ===============================
score = 0
last_fire = 0
FIRE_RATE = 0.15
next_spawn_score = 10

# ===============================
# Helper: overlay PNG
# ===============================
def overlay_png(img, png, x, y):
    h, w = png.shape[:2]
    ih, iw = img.shape[:2]
    if x < 0 or y < 0 or x + w > iw or y + h > ih:
        return
    alpha = png[:, :, 3] / 255.0
    for c in range(3):
        img[y:y+h, x:x+w, c] = (
            alpha * png[:, :, c] +
            (1 - alpha) * img[y:y+h, x:x+w, c]
        )

print("[INFO] Duck Shoot started")

# ===============================
# Main loop
# ===============================
while True:
    ret, cam = cap.read()
    if not ret:
        break

    cam = cv2.flip(cam, 1)
    frame = bg.copy()

    rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    firing = False
    cx, cy = None, None

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        lm = hand.landmark[9]  # middle finger MCP
        raw_x = int(lm.x * BG_W)
        raw_y = int(lm.y * BG_H)

        if smooth_x is None:
            smooth_x, smooth_y = raw_x, raw_y
        else:
            smooth_x = int(smooth_x * (1 - SMOOTHING) + raw_x * SMOOTHING)
            smooth_y = int(smooth_y * (1 - SMOOTHING) + raw_y * SMOOTHING)

        cx, cy = smooth_x, smooth_y

        # Fire when thumb straight
        thumb_tip = hand.landmark[4]
        thumb_mcp = hand.landmark[2]
        thumb_dist = abs(thumb_tip.x - thumb_mcp.x) + abs(thumb_tip.y - thumb_mcp.y)
        thumb_straight = thumb_dist > 0.12
        index_extended = hand.landmark[8].y < hand.landmark[6].y

        if thumb_straight and index_extended:
            firing = True

    # ===============================
    # Move ducks
    # ===============================
    
    for d in ducks:
      d["x"] += d["vx"]
      d["y"] += d["vy"]

      # Respawn if fully outside screen
      if (
        d["x"] < -DUCK_W - 50 or
        d["x"] > BG_W + 50 or
        d["y"] < -DUCK_H - 50 or
        d["y"] > BG_H + 50
      ):
        d.update(spawn_duck())
        

    # ===============================
    # Shooting
    # ===============================
    if firing and cx is not None:
        now = time.time()
        if now - last_fire > FIRE_RATE:
            last_fire = now
            for d in ducks[:]:
                if d["x"] < cx < d["x"] + DUCK_W and d["y"] < cy < d["y"] + DUCK_H:
                    score += 1
                    ducks.remove(d)
                    ducks.append(spawn_duck())
                    break

    # ===============================
    # Difficulty scaling
    # ===============================
    if score >= next_spawn_score:
        ducks.append(spawn_duck())
        ducks.append(spawn_duck())
        next_spawn_score += 10

    # ===============================
    # Draw ducks
    # ===============================
    for d in ducks:
        overlay_png(frame, duck, int(d["x"]), int(d["y"]))

    # ===============================
    # Draw crosshair
    # ===============================
    if cx is not None:
        overlay_png(
            frame,
            crosshair,
            cx - CROSSHAIR_SIZE // 2,
            cy - CROSSHAIR_SIZE // 2
        )

    # ===============================
    # HUD
    # ===============================
    cv2.putText(
        frame,
        f"Score: {score}",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 0),
        3
    )

    # ===============================
    # Camera PiP
    # ===============================
    cam_small = cv2.resize(cam, (320, 180))
    x1 = BG_W - 340
    y1 = BG_H - 200
    frame[y1:y1+180, x1:x1+320] = cam_small
    cv2.rectangle(frame, (x1, y1), (x1+320, y1+180), (255, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()


