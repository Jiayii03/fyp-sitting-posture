import cv2

def find_available_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

# cameras = find_available_cameras()
# print("Available cameras:", cameras)

# Open the USB webcam (change the index to your USB camera's index, e.g., 1)
camera_index = 1
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Unable to access camera {camera_index}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    cv2.imshow("USB Webcam", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

