import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

zoom_factor = 1.0
prev_time = time.time()
prev_position = None
smoothed_position = None
alpha = 0.6

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    zoom_factor = max(1, min(zoom_factor, 3))

    zoom_width, zoom_height = int(width / zoom_factor), int(height / zoom_factor)
    center_x, center_y = width // 2, height // 2

    x1, y1 = max(0, center_x - zoom_width // 2), max(0, center_y - zoom_height // 2)
    x2, y2 = min(width, center_x + zoom_width // 2), min(height, center_y + zoom_height // 2)

    zoomed_frame = frame[y1:y2, x1:x2]
    zoomed_frame = cv2.resize(zoomed_frame, (width, height))

    gray = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough Circle Transform
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )

    ball_position = None

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Pick largest circle (likely the ball)
        largest_circle = max(circles[0, :], key=lambda c: c[2])
        cx, cy, radius = largest_circle
        ball_position = (cx, cy)

        cv2.circle(zoomed_frame, (cx, cy), radius, (0, 255, 0), 2)
        cv2.circle(zoomed_frame, (cx, cy), 5, (0, 0, 255), -1)

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

        cv2.circle(zoomed_frame, (predicted_x, predicted_y), 8, (255, 0, 0), 2)
        cv2.putText(zoomed_frame, "Predicted", (predicted_x + 10, predicted_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.arrowedLine(zoomed_frame, ball_position, (predicted_x, predicted_y),
                        (255, 255, 0), 2, tipLength=0.3)

    prev_time = current_time
    prev_position = ball_position

    cv2.imshow("Ball Tracking & Prediction", zoomed_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('+'), ord('=')]:
        zoom_factor += 0.1
    elif key == ord('-'):
        zoom_factor -= 0.1

cap.release()
cv2.destroyAllWindows()
