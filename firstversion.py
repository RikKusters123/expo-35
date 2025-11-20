import cv2
import numpy as np
import time

# open the video stream
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

zoom_factor = 1.0

prev_time = time.time()
prev_position = None
smoothed_position = None
alpha = 0.6  # smoothing factor

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    zoom_factor = max(1, min(zoom_factor, 3))

    zoom_width, zoom_height = int(width / zoom_factor), int(height / zoom_factor)
    center_x, center_y = width // 2, height // 2

    x1 = max(0, center_x - zoom_width // 2)
    y1 = max(0, center_y - zoom_height // 2)
    x2 = min(width, center_x + zoom_width // 2)
    y2 = min(height, center_y + zoom_height // 2)

    zoomed_frame = frame[y1:y2, x1:x2]
    zoomed_frame = cv2.resize(zoomed_frame, (width, height))

    gray = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)

    # Morphological cleaning
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 0

    ball_position = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # ignore small contours
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # Shape filtering — can adapt for circle, rectangle, etc.
        if len(approx) >= 8:  # approximate circle/ellipse
            if area > max_area:
                max_area = area
                largest_contour = contour

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cx = x + w // 2
        cy = y + h // 2
        ball_position = (cx, cy)

        (circle_x, circle_y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(circle_x), int(circle_y))
        radius = int(radius)

        cv2.circle(zoomed_frame, center, radius, (0, 255, 0), 2)  # green circle
        cv2.circle(zoomed_frame, center, 5, (0, 0, 255), -1)       # red center dot

    # Velocity & prediction
    current_time = time.time()
    if ball_position and prev_position:
        delta_time = current_time - prev_time
        velocity_x = (ball_position[0] - prev_position[0]) / delta_time
        velocity_y = (ball_position[1] - prev_position[1]) / delta_time

        if smoothed_position:
            smoothed_x = int(alpha * ball_position[0] + (1 - alpha) * smoothed_position[0])
            smoothed_y = int(alpha * ball_position[1] + (1 - alpha) * smoothed_position[1])
            smoothed_position = (smoothed_x, smoothed_y)
        else:
            smoothed_position = ball_position

        prediction_time = 0.1
        predicted_x = int(smoothed_position[0] + velocity_x * prediction_time)
        predicted_y = int(smoothed_position[1] + velocity_y * prediction_time)

        cv2.circle(zoomed_frame, (predicted_x, predicted_y), 8, (255, 0, 0), 2)  # predicted point
        cv2.putText(zoomed_frame, "Predicted", (predicted_x + 10, predicted_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.arrowedLine(zoomed_frame, ball_position, (predicted_x, predicted_y),
                        (255, 255, 0), 2, tipLength=0.3)

    prev_time = current_time
    prev_position = ball_position

    cv2.imshow('Shape Tracking & Prediction', zoomed_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        zoom_factor += 0.1
    elif key == ord('-'):
        zoom_factor -= 0.1

cap.release()
cv2.destroyAllWindows()
