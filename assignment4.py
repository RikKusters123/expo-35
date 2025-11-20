import cv2
import numpy as np

# opens video stream 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# checks if camera works
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

zoom_factor = 1.0  

# detects colors
COLOR_RANGES = {
    "Red": [(0, 120, 70), (10, 255, 255)],  
    "Red": [(170, 120, 70), (180, 255, 255)],  
    "Green": [(40, 40, 40), (90, 255, 255)],
    "Blue": [(90, 50, 50), (130, 255, 255)],
    "Yellow": [(24, 150, 150), (35, 255, 255)] 
}
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # frame height and width
    height, width, _ = frame.shape

    # ZOOM
    zoom_factor = max(1, min(zoom_factor, 3))

    # frame for the zoom
    zoom_width, zoom_height = int(width / zoom_factor), int(height / zoom_factor)
    center_x, center_y = width // 2, height // 2

    x1, y1 = max(0, center_x - zoom_width // 2), max(0, center_y - zoom_height // 2)
    x2, y2 = min(width, center_x + zoom_width // 2), min(height, center_y + zoom_height // 2)

    zoomed_frame = frame[y1:y2, x1:x2]
    zoomed_frame = cv2.resize(zoomed_frame, (width, height))

    # gray scale 
    gray = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2GRAY)
    
    # takes out noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # detects sides
    edges = cv2.Canny(blurred, 50, 150)

    # makes HSV scale
    hsv = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2HSV)

    # detects de contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # draws the contours
        x, y, w, h = cv2.boundingRect(approx)
        shape_roi = hsv[y:y+h, x:x+w]
        
        # finds color 
        color_detected = "Unknown"
        for color, (lower, upper) in COLOR_RANGES.items():
            mask = cv2.inRange(shape_roi, np.array(lower), np.array(upper))
            if cv2.countNonZero(mask) > 500:
                color_detected = color
                break

        # determents the shape
        if len(approx) == 3:
            shape_name = "Triangle"
        elif len(approx) == 4:
            aspect_ratio = float(w) / h
            shape_name = "Square" if 0.9 < aspect_ratio < 1.1 else "Rectangle"
        elif len(approx) > 5:
            shape_name = "Circle"
        else:
            shape_name = "Unknown"

        # finds the center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(zoomed_frame, (cx, cy), 5, (0, 0, 255), -1)

        # draws detected shape
        cv2.drawContours(zoomed_frame, [approx], 0, (0, 255, 0), 2)
        cv2.putText(zoomed_frame, f"{shape_name}, {color_detected}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # projects the frame
    cv2.imshow('Shape & Color Detection', zoomed_frame)

    # command keys for quiting and zooming
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('+') or key == ord('='):  # Zoom In
        zoom_factor += 0.1
    elif key == ord('-'):  # Zoom Out
        zoom_factor -= 0.1

# clears all
cap.release()
cv2.destroyAllWindows()
