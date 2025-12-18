from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time
import os

# ==============================
# ⚙️ CONFIGURATION
# ==============================

MODEL_FILENAME = "liveness.model" 

# We check current folder first, then livenessdetect folder
if os.path.exists(MODEL_FILENAME):
    MODEL_PATH = MODEL_FILENAME
elif os.path.exists(os.path.join("livenessdetect", MODEL_FILENAME)):
    MODEL_PATH = os.path.join("livenessdetect", MODEL_FILENAME)
else:
    # Fallback to the explicit path if needed
    MODEL_PATH = MODEL_FILENAME

# ==============================
# ✅ Load Model and Face Detector
# ==============================
try:
    print(f"🔄 Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"   Make sure '{MODEL_FILENAME}' is in the same folder as this script.")
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

    print("✅ Webcam started. Press 'q' to quit.")
    
    prev_label = ""
    label_display_time = 0
    display_duration = 1.5 

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("⚠️ Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw main guide box
        cv2.rectangle(frame, (400, 100), (900, 550), (255, 0, 0), 2)
        cv2.putText(frame, "Keep face inside blue box", (430, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        faces_inside_box = []
        for (x, y, w, h) in faces:
            # Check if face is roughly inside the guide box
            if 400 < x < 800 and 100 < y < 300 and 400 < (x + w) < 900:
                faces_inside_box.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Handle detection logic
        if len(faces_inside_box) > 1:
            cv2.putText(frame, "⚠️ Multiple Faces!", (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        elif len(faces_inside_box) == 1:
            (x, y, w, h) = faces_inside_box[0]
            
            # Preprocess
            face_img = frame[y:y + h, x:x + w]
            try:
                face_img = cv2.resize(face_img, (128, 128))
                face_img = face_img.astype("float") / 255.0
                face_img = img_to_array(face_img)
                face_img = np.expand_dims(face_img, axis=0)

                # Predict
                preds = model.predict(face_img, verbose=0)[0]
                
                # Based on your training.py: Index 0=Real, Index 1=Fake
                # (because 'fake' comes before 'real' in folder names usually, 
                # BUT your code manually set 'fake'=1, 'real'=0)
                # So: Index 0 = Real, Index 1 = Fake.
                (real, fake) = preds
                
                label = "REAL" if real > fake else "FAKE"
                color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
                
                # Show confidence score
                score = max(real, fake) * 100
                display_text = f"{label}: {score:.2f}%"

                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

            except Exception as e:
                pass # Skip frames where resizing fails (e.g. face partially out of view)

        else:
            cv2.putText(frame, "Place face in box", (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Face Liveness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    predictperson()