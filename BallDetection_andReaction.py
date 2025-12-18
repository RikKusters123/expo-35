import cv2
import numpy as np
import time

# === 1. CONFIGURATION & STYLING ===
num_rods = 4
players_per_rod = [3, 2, 3, 1]
rod_color = (0, 255, 255)
player_color = (0, 128, 255)
active_rod_color = (0, 0, 255)
proximity_color = (128, 0, 128)
player_radius = 12
rod_thickness = 2
move_speed = 12

# Prediction Constants
PREDICTION_LOOKAHEAD = 8  # How many frames into the future to predict
VELOCITY_SMOOTHING = 0.7  # 0 to 1 (higher = less jitter, but slower reaction)

# Tracking & Calibration Constants
SCAN_INTERVAL = 0.3
PROXIMITY_THRESHOLD = 55
BALL_RADIUS_DEFAULT = 20
CALIBRATION_FRAMES = 30

# === 2. STATE VARIABLES ===
last_scan_time = 0
tracking_active = False
tracker = None
calibrated = False
initialized = False
radius_measurements = []
ball_radius = BALL_RADIUS_DEFAULT
rods = []

# Velocity/Prediction variables
prev_ball_pos = None
current_velocity = np.array([0.0, 0.0])


# === 3. HELPER FUNCTIONS ===
def create_tracker():
    return cv2.TrackerKCF_create()


def initialize_table(width, height):
    global rods
    rods = []
    spacing = width // (num_rods + 1)
    for i in range(1, num_rods + 1):
        x = i * spacing
        num_players = players_per_rod[i - 1]
        y_spacing = height // (num_players + 1)
        player_positions = [[x, j * y_spacing] for j in range(1, num_players + 1)]
        top_y = min(p[1] for p in player_positions)
        bottom_y = max(p[1] for p in player_positions)

        rods.append({
            "x": x,
            "y_offset": 0,
            "players": player_positions,
            "min_offset": -(top_y - player_radius),
            "max_offset": height - bottom_y - player_radius
        })


# === 4. CAMERA INITIALIZATION ===
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# === 5. MAIN LOOP ===
while True:
    ret, frame = cap.read()
    if not ret: break
    height, width, _ = frame.shape
    current_time = time.time()

    if not initialized:
        initialize_table(width, height)
        initialized = True

    ball_position = None
    predicted_position = None
    detected_radius = ball_radius

    # === STATE MACHINE: TRACKING VS SCANNING ===
    if tracking_active and (current_time - last_scan_time < SCAN_INTERVAL):
        success, bbox = tracker.update(frame)
        if success:
            tx, ty, tw, th = [int(v) for v in bbox]
            ball_position = (tx + tw // 2, ty + th // 2)
            detected_radius = tw // 2
        else:
            tracking_active = False
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 50, param1=50, param2=35, minRadius=10, maxRadius=60)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            largest = max(circles[0, :], key=lambda c: c[2])
            cx, cy, r = largest
            ball_position = (cx, cy)
            detected_radius = r
            if not calibrated:
                radius_measurements.append(r)
                if len(radius_measurements) >= CALIBRATION_FRAMES:
                    ball_radius = int(np.mean(radius_measurements))
                    calibrated = True
            tracker = create_tracker()
            bbox = (cx - r, cy - r, r * 2, r * 2)
            tracker.init(frame, bbox)
            tracking_active = True
            last_scan_time = current_time

    # === PREDICTION LOGIC ===
    if ball_position is not None:
        curr_p = np.array(ball_position)
        if prev_ball_pos is not None:
            # Calculate instantaneous velocity
            inst_velocity = curr_p - np.array(prev_ball_pos)
            # Smooth the velocity to prevent erratic rod movement
            current_velocity = (current_velocity * VELOCITY_SMOOTHING) + (inst_velocity * (1 - VELOCITY_SMOOTHING))

            # Project position into the future
            pred_x = int(curr_p[0] + current_velocity[0] * PREDICTION_LOOKAHEAD)
            pred_y = int(curr_p[1] + current_velocity[1] * PREDICTION_LOOKAHEAD)
            predicted_position = (pred_x, pred_y)

        prev_ball_pos = ball_position
    else:
        prev_ball_pos = None  # Reset if ball is lost

    # === ROD MOVEMENT LOGIC (Based on Prediction) ===
    active_rod = None
    # If we have a prediction, use it; otherwise use current position
    target_pos = predicted_position if predicted_position else ball_position

    if target_pos:
        tx, ty = target_pos
        # Find rod closest to the CURRENT ball position (for logic)
        bx, by = ball_position if ball_position else target_pos
        active_rod = min(rods, key=lambda r: abs(bx - r["x"]))

        # Move active rod towards PREDICTED y-coordinate
        player_abs_y = [p[1] + active_rod["y_offset"] for p in active_rod["players"]]
        closest_player_y = min(player_abs_y, key=lambda py: abs(py - ty))

        diff = ty - closest_player_y
        delta = diff if abs(diff) <= move_speed else (move_speed if diff > 0 else -move_speed)
        new_offset = active_rod["y_offset"] + delta
        active_rod["y_offset"] = max(active_rod["min_offset"], min(new_offset, active_rod["max_offset"]))

    # === DRAWING PHASE ===
    for rod in rods:        
        x, y_off = int(rod["x"]), int(rod["y_offset"])
        r_color = active_rod_color if rod is active_rod else rod_color
        cv2.line(frame, (x, 0), (x, height), r_color, rod_thickness)

        for px, py in rod["players"]:
            py_disp = int(py + y_off)
            current_p_color = player_color
            if ball_position:
                dist = np.sqrt((ball_position[0] - x) ** 2 + (ball_position[1] - py_disp) ** 2)
                if dist < PROXIMITY_THRESHOLD: current_p_color = proximity_color

            cv2.circle(frame, (x, py_disp), player_radius, current_p_color, -1)
            cv2.circle(frame, (x, py_disp), player_radius, (0, 0, 0), 2)

    # Draw Prediction Arrow
    if ball_position and predicted_position:
        # Draw Arrow
        cv2.arrowedLine(frame, ball_position, predicted_position, (0, 255, 0), 3, tipLength=0.3)
        # Draw Predicted "Ghost" Ball
        cv2.circle(frame, predicted_position, 5, (0, 255, 0), -1)

    # === UI OVERLAY ===
    status_text = "LOCKED" if tracking_active else "SCANNING..."
    cv2.putText(frame, f"Mode: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if tracking_active else (0, 255, 255), 2)

    if ball_position:
        cv2.circle(frame, ball_position, detected_radius, (255, 255, 255), 2)

    cv2.imshow("Foosball Prediction System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
