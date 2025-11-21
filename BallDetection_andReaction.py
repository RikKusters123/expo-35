import cv2
import numpy as np
import time

# === CAMERA INITIALIZATION ===
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# === FOOSBALL TABLE CONFIGURATION ===
num_rods = 4                                 # Number of vertical rods (columns of players)
players_per_rod = [3, 2, 3, 1]               # Number of players on each rod
rod_color = (0, 255, 255)                    # Yellow color for normal rods
player_color = (0, 128, 255)                 # Orange color for players
active_color = (0, 0, 255)                   # Red color for the currently active rod
player_radius = 12                           # Radius of each player circle
rod_thickness = 2                            # Thickness of rod line
move_speed = 10                              # Speed (pixels/frame) for vertical movement
initialized = False                          # Flag to check if setup is done

# Storage for all rod information
rods = []

# === BALL DETECTION PARAMETERS ===
BALL_DIAMETER_DEFAULT = 40       # Default guess (pixels)
BALL_TOLERANCE = 10              # Allowed variation for detection
BALL_RADIUS = BALL_DIAMETER_DEFAULT // 2
calibration_frames = 30          # Number of frames to use for auto-calibration
radius_measurements = []         # Store detected radii for calibration
calibrated = False               # Flag to indicate if calibration finished


# === FUNCTION: Initialize the Foosball Table ===
def initialize_table(width, height):
    """Create rod positions and evenly distribute players across each rod."""
    global rods
    rods = []
    spacing = width // (num_rods + 1)

    for i in range(1, num_rods + 1):
        x = i * spacing  # X position of rod
        num_players = players_per_rod[i - 1]
        y_spacing = height // (num_players + 1)

        # Player (x, y) coordinates
        player_positions = [[x, j * y_spacing] for j in range(1, num_players + 1)]

        rods.append({
            "x": x,
            "y_offset": 0,
            "players": player_positions,
            "min_offset": 0,
            "max_offset": 0
        })


# === FUNCTION: Set Vertical Movement Limits for Each Rod ===
def update_rod_limits(rod, height):
    """Ensure players stay visible by limiting rod motion."""
    base_positions = [p[1] for p in rod["players"]]
    top_y = min(base_positions)
    bottom_y = max(base_positions)

    rod["min_offset"] = -(top_y - player_radius)
    rod["max_offset"] = height - bottom_y - player_radius


# === TRACKING SETUP ===
prev_time = time.time()
prev_position = None

# === MAIN LOOP ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Initialize table layout once
    if not initialized:
        initialize_table(width, height)
        for rod in rods:
            update_rod_limits(rod, height)
        initialized = True

    # === BALL DETECTION ===
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles (potential balls)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1.2, 50,
        param1=50, param2=30,
        minRadius=max(5, BALL_RADIUS - BALL_TOLERANCE),
        maxRadius=BALL_RADIUS + BALL_TOLERANCE
    )

    ball_position = None
    detected_radius = None

    if circles is not None:
        # Select largest circle (most likely the ball)
        circles = np.uint16(np.around(circles))
        largest = max(circles[0, :], key=lambda c: c[2])
        cx, cy, r = largest
        ball_position = (cx, cy)
        detected_radius = r

        # Draw the ball on the frame
        cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # === AUTO-CALIBRATION ===
        if not calibrated and len(radius_measurements) < calibration_frames:
            radius_measurements.append(r)
        elif not calibrated and len(radius_measurements) >= calibration_frames:
            # Compute average radius over calibration frames
            avg_radius = int(np.mean(radius_measurements))
            BALL_RADIUS = avg_radius
            calibrated = True
            print(f"[INFO] Ball size calibrated: {BALL_RADIUS * 2} px diameter")

    # === DETERMINE CLOSEST ROD TO BALL ===
    active_rod = None
    if ball_position:
        bx, by = ball_position
        closest_x_distance = float('inf')

        for rod in rods:
            dist_x = abs(bx - rod["x"])
            if dist_x < closest_x_distance:
                closest_x_distance = dist_x
                active_rod = rod

    # === MOVE THE ACTIVE ROD TOWARD THE BALL ===
    if active_rod and ball_position:
        bx, by = ball_position
        player_abs_positions = [p[1] + active_rod["y_offset"] for p in active_rod["players"]]
        closest_player_y = min(player_abs_positions, key=lambda py: abs(py - by))
        closest_diff = by - closest_player_y

        # Move rod step-by-step toward the ball
        if abs(closest_diff) > move_speed:
            delta = move_speed if closest_diff > 0 else -move_speed
        else:
            delta = closest_diff  # Snap exactly when close enough

        # Enforce movement limits
        new_offset = active_rod["y_offset"] + delta
        new_offset = max(active_rod["min_offset"], min(new_offset, active_rod["max_offset"]))
        active_rod["y_offset"] = new_offset

    # === DRAW RODS AND PLAYERS ===
    for rod in rods:
        x = int(rod["x"])
        y_offset = int(rod["y_offset"])
        color = active_color if rod is active_rod else rod_color

        # Draw rod
        cv2.line(frame, (x, 0), (x, height), color, rod_thickness)

        # Draw players
        for px, py in rod["players"]:
            py_disp = int(py + y_offset)
            cv2.circle(frame, (x, py_disp), player_radius, player_color, -1)
            cv2.circle(frame, (x, py_disp), player_radius, (0, 0, 0), 2)

    # === DISPLAY CALIBRATION STATUS ===
    if not calibrated:
        cv2.putText(frame, "Calibrating ball size...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Ball diameter: {BALL_RADIUS * 2}px", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # === SHOW FRAME ===
    cv2.imshow("Foosball Simulation", frame)

    # Quit on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()
