import cv2

# ----------------------
# CONFIGURATION
# ----------------------
# Set your camera resolution
CAM_WIDTH = 1920
CAM_HEIGHT = 1080

# Distance from camera to bottom of the football table (in mm or cm)
# You can change this value during calibration
camera_to_table_distance = 500  # example: 500 mm

# ----------------------
# CAMERA SETUP
# ----------------------
cap = cv2.VideoCapture(1)  # 0 is the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# ----------------------
# CALIBRATION FUNCTION
# ----------------------
def draw_roi(event, x, y, flags, param):
    """Mouse callback to draw rectangle ROI"""
    global roi_start, roi_end, drawing, roi_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)
        drawing = False
        roi_defined = True
        print(f"ROI defined from {roi_start} to {roi_end}")

# ----------------------
# MAIN LOOP
# ----------------------
roi_start = (0, 0)
roi_end = (0, 0)
drawing = False
roi_defined = False

cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", draw_roi)

print("Instructions:")
print("1. Use your mouse to draw a rectangle around the table in the video frame.")
print("2. Press 'c' to confirm calibration and save the ROI.")
print("3. Press 'q' to quit without saving.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Draw rectangle while dragging
    if drawing or roi_defined:
        cv2.rectangle(frame, roi_start, roi_end, (0, 255, 0), 2)

    cv2.putText(frame, f"Distance to table: {camera_to_table_distance} mm",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Calibration", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and roi_defined:
        # Save ROI and distance
        roi_data = {
            "roi_start": roi_start,
            "roi_end": roi_end,
            "distance_mm": camera_to_table_distance
        }
        import json
        with open("calibration.json", "w") as f:
            json.dump(roi_data, f, indent=4)
        print("Calibration saved to calibration.json")
        break
    elif key == ord('q'):
        print("Quitting without saving")
        break

cap.release()
cv2.destroyAllWindows()