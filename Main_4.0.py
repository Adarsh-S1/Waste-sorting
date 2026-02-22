import cv2
import numpy as np
import time
import serial
import threading
import queue
from flask import Flask, Response, render_template_string, jsonify
from tflite_runtime.interpreter import Interpreter
from adafruit_servokit import ServoKit

# --- CONFIGURATION ---
MODEL_PATH = 'model_2.tflite'
LABEL_PATH = 'label_2.txt'

CAMERA_WIDTH = 640  # Increased for better crop quality
CAMERA_HEIGHT = 480
FRAME_RATE = 20

INPUT_WIDTH = 224
INPUT_HEIGHT = 224
CONFIDENCE_THRESHOLD = 0.85 # Adjusted slightly for averaged results

NUM_SERVOS = 4
SERVO_ANGLES = {
    'Battery': [90, 90, 90, 180],
    'PCB':     [90, 90, 0, 90],
    'metal':   [0, 90, 90, 90],
    'plastic': [90, 180, 90, 90], 
    'default': [90, 90, 90, 90],
}

SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 9600
# --- END CONFIGURATION ---

# --- GLOBAL STATE ---
task_queue = queue.Queue()
output_frame = None
frame_lock = threading.Lock()

app = Flask(__name__)

# --- UTILITY FUNCTIONS ---

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def center_crop_and_resize(frame, size=(224, 224)):
    """Crops the frame to a square to prevent squishing the object."""
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    return cv2.resize(cropped, size)

def set_servos(kit, angles):
    if kit is None: return
    try:
        for i in range(NUM_SERVOS):
            kit.servo[i].angle = angles[i]
    except Exception as e:
        print(f"Error moving servos: {e}")

def send_to_lcd(ser, line1, line2=""):
    if ser is None: return
    try:
        line1_trunc = line1[:16]
        line2_trunc = line2[:16]
        message = f"{line1_trunc}|{line2_trunc}\n"
        ser.write(message.encode('utf-8'))
    except Exception as e:
        print(f"LCD Error: {e}")

# --- WORKER THREADS ---

def inference_worker():
    print("[Thread] Inference Worker started...")
    kit = None
    ser = None
    interpreter = None
    labels = []

    try:
        kit = ServoKit(channels=16)
    except Exception as e:
        print(f"[Warning] ServoKit init failed: {e}")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) 
        send_to_lcd(ser, "System Ready", "Web Connected")
    except Exception as e:
        print(f"[Warning] Serial/LCD init failed: {e}")
    
    try:
        labels = load_labels(LABEL_PATH)
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        print(f"[CRITICAL] Model Loading Error: {e}")

    if kit:
        set_servos(kit, SERVO_ANGLES['default'])

    while True:
        command, frame, response_queue = task_queue.get()
        
        if command == 'SORT' and interpreter:
            all_scores = []
            
            # --- MULTI-FRAME AVERAGING ---
            # We take 3 quick snapshots to ensure lighting/blur doesn't ruin the result
            for _ in range(3):
                with frame_lock:
                    current_f = output_frame.copy() if output_frame is not None else None
                
                if current_f is not None:
                    # 1. Pre-process (RGB + Square Crop)
                    rgb = cv2.cvtColor(current_f, cv2.COLOR_BGR2RGB)
                    processed = center_crop_and_resize(rgb, (INPUT_WIDTH, INPUT_HEIGHT))
                    
                    # 2. Add Batch Dimension
                    input_data = np.expand_dims(processed, axis=0)

                    # 3. TM NORMALIZATION: Scale [0,255] to [-1, 1]
                    if input_details[0]['dtype'] == np.float32:
                        input_data = (input_data.astype(np.float32) / 127.5) - 1.0

                    # 4. Inference
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    all_scores.append(interpreter.get_tensor(output_details[0]['index'])[0])
                
                time.sleep(0.05) # Tiny gap between snapshots

            if all_scores:
                # Average the results across frames
                avg_scores = np.mean(all_scores, axis=0)
                class_id = np.argmax(avg_scores)
                confidence = avg_scores[class_id]
                class_name = labels[class_id]

                if confidence > CONFIDENCE_THRESHOLD:
                    result_text = f"{class_name} ({confidence:.0%})"
                    send_to_lcd(ser, f"Found: {class_name}", f"Conf:{confidence:.0%}")
                    if class_name in SERVO_ANGLES:
                        set_servos(kit, SERVO_ANGLES[class_name])
                else:
                    result_text = "Not Found"
                    send_to_lcd(ser, "Low Confidence", "Try Again")
            else:
                result_text = "Cam Error"

            if response_queue:
                response_queue.put(result_text)

        elif command == 'RESET':
            if kit: set_servos(kit, SERVO_ANGLES['default'])
            send_to_lcd(ser, "Status: Ready", "Waiting...")
            if response_queue: response_queue.put("Reset Complete")

        task_queue.task_done()

def camera_thread():
    global output_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                output_frame = frame.copy()
        time.sleep(0.01)

# --- FLASK APPLICATION ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>E-Waste Sorter Pro</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Segoe UI', sans-serif; text-align: center; background: #1a1a1a; color: #fff; margin: 0; padding: 20px; }
        .container { margin-top: 20px; }
        img { border: 4px solid #333; border-radius: 12px; max-width: 100%; height: auto; }
        .status-box { background: #262626; padding: 20px; margin: 20px auto; max-width: 500px; border-radius: 10px; border-left: 5px solid #00e676; }
        h2 { margin: 5px 0; color: #00e676; }
        .btn { background: #008CBA; color: #fff; padding: 15px 40px; font-size: 18px; margin: 10px; cursor: pointer; border: none; border-radius: 8px; font-weight: bold; }
        .btn-red { background: #d32f2f; }
        .loading { color: #ffeb3b; }
    </style>
</head>
<body>
    <h1>♻️ E-Waste Smart Sorter</h1>
    <div><img src="{{ url_for('video_feed') }}"></div>
    <div class="status-box">
        <small style="color: #888;">AI CLASSIFICATION</small>
        <h2 id="status-text">System Ready</h2>
    </div>
    <div class="container">
        <button class="btn" onclick="triggerSort()">SCAN OBJECT</button>
        <button class="btn btn-red" onclick="triggerReset()">RESET</button>
    </div>
    <script>
        function triggerSort() {
            const st = document.getElementById("status-text");
            st.innerText = "Analyzing... 🔍";
            st.className = "loading";
            fetch('/api/sort', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    st.innerText = data.result;
                    st.className = "";
                    st.style.color = data.result.includes("Found") ? "#00e676" : "#ff5252";
                });
        }
        function triggerReset() {
            fetch('/api/reset', { method: 'POST' });
            document.getElementById("status-text").innerText = "System Ready";
            document.getElementById("status-text").style.color = "#00e676";
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

def generate_frames():
    while True:
        with frame_lock:
            if output_frame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/sort', methods=['POST'])
def api_sort():
    res_q = queue.Queue()
    task_queue.put(('SORT', None, res_q))
    try:
        return jsonify({"result": res_q.get(timeout=6)})
    except:
        return jsonify({"result": "Timeout Error"}), 500

@app.route('/api/reset', methods=['POST'])
def api_reset():
    task_queue.put(('RESET', None, None))
    return jsonify({"status": "Reset triggered"})

if __name__ == '__main__':
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=inference_worker, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)