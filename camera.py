import cv2

cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame…")
        break

    cv2.imshow("Pi Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
