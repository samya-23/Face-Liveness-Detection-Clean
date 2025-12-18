import cv2
import os
import random

# 1. Initialize the camera
video_capture = cv2.VideoCapture(0)

# 2. Setup the save path automatically
# This points to: livenessdetect/dataset/fake/
save_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "fake")

# Create the folder if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 3. Load Face Detector
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print(f"Saving images to: {save_path}")
print("Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    
    # Draw the guide box
    cv2.rectangle(frame, (400, 100), (900, 550), (255,0,0), 2)
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        var1 = frame.copy()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print(x, y, x+w, y+h)
        
        # Check if face is inside the guide box
        if(x<800 and x>400 and y<300 and y>100 and (x+w)<900 and (x+w)>400 and (y+h)<560 and len(faces)==1):
            
            cv2.putText(frame, "Perfect", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save the image
            filename = "fake%d.jpg" % random.randint(40000, 90000)
            full_path = os.path.join(save_path, filename)
            cv2.imwrite(full_path, var1)
            print(f"Saved: {filename}")

    label = "{}".format(len(faces))
    cv2.imshow('Video', frame)
    
    # On pressing the 'q' button the frame capturing will end.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()