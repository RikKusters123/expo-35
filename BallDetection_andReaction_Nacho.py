import cv2
import numpy as np
import serial
import time

# ------------------ USB CONFIG ------------------
TEENSY_PORT = '/dev/ttyACM0'  # adjust to your Teensy port
TEENSY_BAUD = 115200
ser = serial.Serial(TEENSY_PORT, TEENSY_BAUD, timeout=0.1)

# ------------------ Foosball Table Config ------------------
num_rods = 4
players_per_rod = [3, 2, 3, 1]

# Rod offsets for motor mapping
# Assuming motors 1-4 = translation (bars 1-4), 5-8 = rotation (bars 1-4)
def bar_to_motor(bar_index, axis):
    if axis == 'T':  # translation
        return bar_index + 1
    if axis == 'R':  # rotation
        return bar_index + 5

# Rod positions and movement limits
class Rod:
    def __init__(self, x, players, min_offset, max_offset):
        self.x = x
        self.players = players
        self.y_offset = 0
        self.min_offset = min_offset
        self.max_offset = max_offset

rods = []

# ------------------ TABLE INITIALIZATION ------------------
def initialize_table(width, height):
    global rods
    spacing = width // (num_rods + 1)
    for i in range(1, num_rods + 1):
        x = i * spacing
        num_players = players_per_rod[i - 1]
        y_spacing = height // (num_players + 1)
        player_positions = [[x, j * y_spacing] for j in range(1, num_players + 1)]
        top_y = min(p[1] for p in player_positions)
        bottom_y = max(p[1] for p in player_positions)

        rods.append(Rod(x, player_positions, -(top_y-12), height-bottom_y-12))

# ------------------ BALL TRACKING ------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

tracker = None
tracking_active = False
prev_ball_pos = None
current_velocity = np.array([0.0,0.0])
PREDICTION_LOOKAHEAD = 8
VELOCITY_SMOOTHING = 0.7
SCAN_INTERVAL = 0.3
last_scan_time = 0
BALL_RADIUS_DEFAULT = 20
calibrated = False
radius_measurements = []
CALIBRATION_FRAMES = 30
ball_radius = BALL_RADIUS_DEFAULT
initialized = False

def create_tracker():
    return cv2.TrackerKCF_create()

# ------------------ COMMAND SENDING ------------------
def send_motor_command(bar_id, axis, position):
    msg = f"{bar_id},{axis},{position}\n"
    ser.write(msg.encode())

def send_all_rods():
    for idx, rod in enumerate(rods):
        # Send translation motor
        send_motor_command(idx+1, 'T', int(rod.y_offset))
        # Optionally send rotation motor (example: rotation = 0 here)
        send_motor_command(idx+1, 'R', 0)

# ------------------ MAIN LOOP ------------------
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

    # --- TRACKING LOGIC ---
    if tracking_active and (current_time - last_scan_time < SCAN_INTERVAL):
        success, bbox = tracker.update(frame)
        if success:
            tx, ty, tw, th = [int(v) for v in bbox]
            ball_position = (tx+tw//2, ty+th//2)
            detected_radius = tw//2
        else:
            tracking_active = False
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9,9),2)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1.2,50,param1=50,param2=35,minRadius=10,maxRadius=60)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            largest = max(circles[0,:], key=lambda c: c[2])
            cx, cy, r = largest
            ball_position = (cx, cy)
            detected_radius = r
            if not calibrated:
                radius_measurements.append(r)
                if len(radius_measurements) >= CALIBRATION_FRAMES:
                    ball_radius = int(np.mean(radius_measurements))
                    calibrated = True
            tracker = create_tracker()
            bbox = (cx-r, cy-r, r*2, r*2)
            tracker.init(frame, bbox)
            tracking_active = True
            last_scan_time = current_time

    # --- PREDICTION ---
    if ball_position is not None:
        curr_p = np.array(ball_position)
        if prev_ball_pos is not None:
            inst_velocity = curr_p - np.array(prev_ball_pos)
            current_velocity = current_velocity*VELOCITY_SMOOTHING + inst_velocity*(1-VELOCITY_SMOOTHING)
            pred_x = int(curr_p[0] + current_velocity[0]*PREDICTION_LOOKAHEAD)
            pred_y = int(curr_p[1] + current_velocity[1]*PREDICTION_LOOKAHEAD)
            predicted_position = (pred_x, pred_y)
        prev_ball_pos = ball_position
    else:
        prev_ball_pos = None

    # --- ROD CONTROL ---
    target_pos = predicted_position if predicted_position else ball_position
    if target_pos:
        tx, ty = target_pos
        active_rod = min(rods, key=lambda r: abs(tx - r.x))
        player_abs_y = [p[1] + active_rod.y_offset for p in active_rod.players]
        closest_player_y = min(player_abs_y, key=lambda py: abs(py-ty))
        diff = ty - closest_player_y
        delta = diff if abs(diff)<=move_speed else (move_speed if diff>0 else -move_speed)
        new_offset = active_rod.y_offset + delta
        active_rod.y_offset = max(active_rod.min_offset, min(new_offset, active_rod.max_offset))

    # --- SEND TO TEENSY ---
    send_all_rods()

    # --- DRAWING (optional) ---
    for rod in rods:
        x, y_off = int(rod.x), int(rod.y_offset)
        cv2.line(frame, (x,0),(x,height),(0,255,255),2)
        for px, py in rod.players:
            py_disp = int(py + y_off)
            cv2.circle(frame,(x,py_disp),12,(0,128,255),-1)
    if ball_position:
        cv2.circle(frame, ball_position, detected_radius, (255,255,255),2)

    cv2.imshow("Foosball Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
