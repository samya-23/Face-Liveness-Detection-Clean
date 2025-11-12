from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import time
import os

# ==============================
# ✅ Load Model and Face Detector
# ==============================
MODEL_PATH = os.path.join("livenessdetect", "models", "anandfinal.hdf5")

try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==============================
# 🎥 Liveness Detection Function
# ==============================
def predictperson():
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("✅ Press 'q' to quit.")
    prev_label = ""
    label_display_time = 0
    display_duration = 1.5  # seconds
    # cv2.namedWindow("Face Liveness Detection", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Face Liveness Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("⚠️ Could not read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw main guide box
        cv2.rectangle(frame, (400, 100), (900, 550), (255, 0, 0), 2)
        cv2.putText(frame,
                    "Keep your head inside the blue box (1 face only)",
                    (30, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        faces_inside_box = []
        for (x, y, w, h) in faces:
            if 400 < x < 800 and 100 < y < 300 and 400 < (x + w) < 900 and 100 < (y + h) < 560:
                faces_inside_box.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Handle multiple or no face cases
        if len(faces_inside_box) > 1:
            cv2.putText(frame, "⚠️ Multiple Faces Detected!",
                        (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        elif len(faces_inside_box) == 1:
            (x, y, w, h) = faces_inside_box[0]
            face_img = frame[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (128, 128))
            face_img = face_img.astype("float") / 255.0
            face_img = img_to_array(face_img)
            face_img = np.expand_dims(face_img, axis=0)

            # Predict
            (real, fake) = model.predict(face_img)[0]
            label = "REAL" if real > fake else "FAKE"
            color = (0, 255, 0) if label == "REAL" else (0, 0, 255)

            # Smooth display
            if label != prev_label or (time.time() - label_display_time) > display_duration:
                label_display_time = time.time()
                prev_label = label

            cv2.putText(frame, f"Result: {label}",
                        (40, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)

        else:
            cv2.putText(frame, "Position your face inside the box",
                        (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Display window
        cv2.imshow("Face Liveness Detection", frame)

        # Quit condition
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# ==============================
# 🚀 Run Main
# ==============================
if __name__ == '__main__':
    predictperson()

