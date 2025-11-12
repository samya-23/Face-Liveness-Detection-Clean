from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2, os, time, sqlite3
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from datetime import datetime

app = Flask(__name__)

# -------- CONFIG --------
MODEL_PATH = os.path.join("livenessdetect", "models", "anandfinal.hdf5")
model = load_model(MODEL_PATH)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

os.makedirs("static/snapshots", exist_ok=True)
os.makedirs("database", exist_ok=True)

# global streaming flag
STREAMING = False

# Robust decision params
N_CONS_FRAMES = 6            # require N consecutive frames to confirm a label
CONF_MARGIN = 0.12           # require difference between probs > margin
LOG_DEBOUNCE_SECONDS = 3.0  # min seconds between logs for same label

# logging DB
def init_db():
    conn = sqlite3.connect("database/logs.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  result TEXT,
                  image_path TEXT)''')
    conn.commit()
    conn.close()

init_db()

def log_result(result, frame):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    img_path = f"static/snapshots/{ts}.jpg"
    cv2.imwrite(img_path, frame)
    conn = sqlite3.connect("database/logs.db")
    c = conn.cursor()
    c.execute("INSERT INTO logs (timestamp, result, image_path) VALUES (?, ?, ?)",
              (ts, result, img_path))
    conn.commit()
    rowid = c.lastrowid
    conn.close()
    return {"id": rowid, "timestamp": ts, "result": result, "image_path": img_path}

# -------- VIDEO/DETECTION GENERATOR --------
def generate_frames():
    """
    - When STREAMING is False: yield a placeholder (camera OFF).
    - When STREAMING is True: open the camera, process frames, confirm label robustly.
      When label is confirmed: log_result AND set STREAMING=False (auto off).
    """
    global STREAMING
    last_logged_label = None
    last_logged_time = 0

    while True:
        if not STREAMING:
            # placeholder until Unlock
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Locked - press Unlock", (40, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
            ret, buf = cv2.imencode('.jpg', placeholder)
            frame_bytes = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.15)
            continue

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            err = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(err, "Error: cannot open camera", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            ret, buf = cv2.imencode('.jpg', err)
            frame_bytes = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5)
            cap.release()
            continue

        cons_real = 0
        cons_fake = 0
        prev_confirmed_label = None

        while STREAMING:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # central alignment box (responsive)
            box_w = int(w * 0.5)
            box_h = int(h * 0.65)
            box_x1 = (w - box_w) // 2
            box_y1 = (h - box_h) // 2
            box_x2 = box_x1 + box_w
            box_y2 = box_y1 + box_h

            # outer border + center box + instruction
            cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (195, 245, 255), 3)
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 3)
            cv2.putText(frame, "Align your face inside the box and hold still",
                        (int(w * 0.05), int(h * 0.08)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 220, 220), 2)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.15, 5, minSize=(int(h*0.12), int(h*0.12)))

            faces_in_box = []
            for (x, y, fw, fh) in faces:
                cx = x + fw // 2
                cy = y + fh // 2
                if box_x1 < cx < box_x2 and box_y1 < cy < box_y2:
                    faces_in_box.append((x, y, fw, fh))
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            confirmed_label = None
            if len(faces_in_box) == 1:
                x, y, fw, fh = faces_in_box[0]
                try:
                    face_resized = cv2.resize(frame[y:y+fh, x:x+fw], (128, 128))
                except Exception:
                    face_resized = None

                if face_resized is not None:
                    img = face_resized.astype("float") / 255.0
                    img = img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    preds = model.predict(img, verbose=0)[0]
                    # assumes ordering: [real_prob, fake_prob]
                    real_prob, fake_prob = float(preds[0]), float(preds[1])

                    if real_prob - fake_prob > CONF_MARGIN:
                        cons_real += 1
                        cons_fake = 0
                    elif fake_prob - real_prob > CONF_MARGIN:
                        cons_fake += 1
                        cons_real = 0
                    else:
                        cons_real = cons_fake = 0

                    cv2.putText(frame, f"R:{real_prob:.2f} F:{fake_prob:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if cons_real >= N_CONS_FRAMES:
                        confirmed_label = "REAL"
                    elif cons_fake >= N_CONS_FRAMES:
                        confirmed_label = "FAKE"
            else:
                cons_real = cons_fake = 0
                if len(faces_in_box) > 1:
                    cv2.putText(frame, "Multiple faces inside box", (int(w*0.2), int(h*0.92)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "No face detected inside the box", (int(w*0.28), int(h*0.92)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # when a confirmed_label is found, log + auto-stop streaming
            if confirmed_label is not None and confirmed_label != prev_confirmed_label:
                now = time.time()
                if last_logged_label != confirmed_label or (now - last_logged_time) > LOG_DEBOUNCE_SECONDS:
                    # log snapshot (synchronous) and set last logged
                    log_result(confirmed_label, frame)
                    last_logged_label = confirmed_label
                    last_logged_time = now

                # show big status then auto stop
                prev_confirmed_label = confirmed_label
                # draw big status immediately
                color = (0,255,0) if confirmed_label == "REAL" else (0,0,255)
                cv2.putText(frame, f"{confirmed_label}", (int(w*0.42), int(h*0.9)),
                            cv2.FONT_HERSHEY_TRIPLEX, 1.4, color, 3)

                # yield the final frame so client gets visual confirmation
                ret2, buf = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

                # auto-stop streaming (camera off)
                STREAMING = False
                break

            # show last confirmed label if exists (keeps overlay)
            if prev_confirmed_label is not None:
                color = (0,255,0) if prev_confirmed_label == "REAL" else (0,0,255)
                cv2.putText(frame, f"{prev_confirmed_label}", (int(w*0.42), int(h*0.9)),
                            cv2.FONT_HERSHEY_TRIPLEX, 1.4, color, 3)

            # encode and yield normal frame
            ret2, buf = cv2.imencode('.jpg', frame)
            frame_bytes = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # release when STREAMING becomes False
        cap.release()
        # reset counters to start clean upon next start
        cons_real = cons_fake = 0
        prev_confirmed_label = None
        time.sleep(0.15)

# -------- ROUTES --------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global STREAMING
    STREAMING = True
    return jsonify({"status": "started"})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global STREAMING
    STREAMING = False
    return jsonify({"status": "stopped"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/last_log')
def last_log():
    """Return latest log entry as JSON (or empty) so frontend can poll."""
    conn = sqlite3.connect("database/logs.db")
    c = conn.cursor()
    c.execute("SELECT id, timestamp, result, image_path FROM logs ORDER BY id DESC LIMIT 1")
    r = c.fetchone()
    conn.close()
    if r:
        return jsonify({"id": r[0], "timestamp": r[1], "result": r[2], "image_path": r[3]})
    else:
        return jsonify({})

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect("database/logs.db")
    c = conn.cursor()
    c.execute("SELECT * FROM logs ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return render_template("dashboard.html", data=data)

@app.route('/delete/<int:rowid>', methods=['POST'])
def delete_log(rowid):
    conn = sqlite3.connect("database/logs.db")
    c = conn.cursor()
    c.execute("SELECT image_path FROM logs WHERE id = ?", (rowid,))
    r = c.fetchone()
    if r and os.path.exists(r[0]):
        os.remove(r[0])
    c.execute("DELETE FROM logs WHERE id = ?", (rowid,))
    conn.commit()
    conn.close()
    return redirect(url_for('dashboard'))

@app.route('/delete_multiple', methods=['POST'])
def delete_multiple():
    # expects JSON with {"ids": [1,2,3]}
    data = request.get_json()
    ids = data.get("ids", [])
    if not ids:
        return jsonify({"status":"error","message":"no ids"}), 400

    conn = sqlite3.connect("database/logs.db")
    c = conn.cursor()
    for _id in ids:
        c.execute("SELECT image_path FROM logs WHERE id = ?", (_id,))
        r = c.fetchone()
        if r and os.path.exists(r[0]):
            os.remove(r[0])
        c.execute("DELETE FROM logs WHERE id = ?", (_id,))
    conn.commit()
    conn.close()
    return jsonify({"status":"ok","deleted":ids})

if __name__ == "__main__":
    app.run(debug=True)
