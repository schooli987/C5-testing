import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

filter_paths = [
    r"C:/Users/LENOVO/Desktop/AI/C4/sunglasses.png",
    r"C:/Users/LENOVO/Desktop/AI/C4/hat.png",
    r"C:/Users/LENOVO/Desktop/AI/C4/moustache.png",
    r"C:/Users/LENOVO/Desktop/AI/C4/crown.png",
]
filter_names = ["Sunglasses", "Hat", "Moustache", "Crown"]

filters = []
for path in filter_paths:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not load {path}")
        exit()
    filters.append(img)

current_filter = 0

def add_filter(frame, overlay_img, x, y, w, h):
    overlay_img = cv2.resize(overlay_img, (w, h))
    b, g, r, a = cv2.split(overlay_img)
    mask = a / 255.0
    for c in range(3):
        frame[y:y+h, x:x+w, c] = (1 - mask) * frame[y:y+h, x:x+w, c] + mask * overlay_img[:, :, c]
    return frame

cap = cv2.VideoCapture(0)

print("Press N to go to next filter, P for previous, Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        name = filter_names[current_filter]

        if name == "Sunglasses":
            fw, fh = w, int(h * 0.3)
            fx, fy = x, y + int(h * 0.3)
        elif name == "Hat" or name == "Crown":
            fw, fh = int(w * 1.2), int(h * 0.6)
            fx, fy = x - int(w * 0.1), y - int(h * 0.6)
        elif name == "Moustache":
            fw, fh = int(w * 0.5), int(h * 0.18)
            fx, fy = x + int(w * 0.25), y + int(h * 0.65)
        else:
            fw, fh = w, int(h * 0.3)
            fx, fy = x, y + int(h * 0.2)

        frame = add_filter(frame, filters[current_filter], fx, fy, fw, fh)
        break

    cv2.putText(frame, filter_names[current_filter], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Fun Filters", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        current_filter = (current_filter + 1) % len(filters)
    elif key == ord('p'):
        current_filter = (current_filter - 1) % len(filters)

cap.release()
cv2.destroyAllWindows()
