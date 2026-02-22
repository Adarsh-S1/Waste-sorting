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
MODEL_PATH = 'model_3.tflite'
LABEL_PATH = 'label_3.txt'

CAMERA_WIDTH = 224
CAMERA_HEIGHT = 224
FRAME_RATE = 20

INPUT_WIDTH = 224
INPUT_HEIGHT = 224
CONFIDENCE_THRESHOLD = 0.90

NUM_SERVOS = 4
SERVO_ANGLES = {
    'Battery': [90, 90, 90, 180],
    'PCB':     [90, 90, 0, 90],
    'metal':   [0, 90, 90, 90],
    'plastic': [90, 180, 90, 90], 
    'default': [90, 90, 90, 90],
}

# *** FIX 1: UPDATED PORT TO MATCH YOUR WORKING TEST ***
SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 9600
# --- END CONFIGURATION ---

# --- GLOBAL STATE ---
task_queue = queue.Queue()
output_frame = None
frame_lock = threading.Lock()

app = Flask(__name__)

# --- HARDWARE FUNCTIONS ---

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

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
        # Added print for debugging
        print(f"LCD Sent: {line1_trunc} / {line2_trunc}")
    except Exception as e:
        print(f"LCD Error: {e}")

# --- WORKER THREADS ---

def inference_worker():
    print("[Thread] Inference Worker started...")
    
    kit = None
    ser = None
    interpreter = None
    labels = []
    input_details = None
    output_details = None

    # Initialize Hardware
    try:
        # 1. Setup Servos
        try:
            kit = ServoKit(channels=16)
        except Exception as e:
            print(f"[Warning] ServoKit init failed: {e}")

        # 2. Setup Serial (LCD)
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2) 
            send_to_lcd(ser, "System Ready", "Web Connected")
        except Exception as e:
            print(f"[Warning] Serial/LCD init failed: {e}")
        
        # 3. Setup Model
        labels = load_labels(LABEL_PATH)
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("[Thread] System Ready.")

    except Exception as e:
        print(f"[CRITICAL] General Init Error: {e}")
        # *** FIX 2: Do NOT return here. Keep thread alive so web doesn't hang. ***

    # Set Initial State
    if kit:
        set_servos(kit, SERVO_ANGLES['default'])

    while True:
        # Wait for command
        command, frame, response_queue = task_queue.get()
        
        if command == 'SORT':
            result_text = "Error"
            
            # Check if model loaded successfully
            if interpreter is None:
                result_text = "Model Error"
                print("Error: Model not loaded, cannot sort.")
            elif frame is not None:
                try:
                    # Pre-process
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_resized = cv2.resize(image_rgb, (INPUT_WIDTH, INPUT_HEIGHT))
                    input_data = np.expand_dims(image_resized, axis=0).astype(input_details[0]['dtype'])

                    # Inference
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    
                    # Result logic
                    scores = output_data[0]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    class_name = labels[class_id]

                    if confidence > CONFIDENCE_THRESHOLD:
                        result_text = f"{class_name} ({confidence:.0%})"
                        print(f"Detected: {result_text}")
                        send_to_lcd(ser, f"Found: {class_name}", f"Conf:{confidence:.0%}")
                        
                        if class_name in SERVO_ANGLES:
                            set_servos(kit, SERVO_ANGLES[class_name])
                        else:
                            print(f"Warning: No servo angle for {class_name}")
                    else:
                        result_text = "Not Found"
                        print("Low confidence - Not Found")
                        send_to_lcd(ser, "Unknown Object", "Try Again")
                except Exception as e:
                    print(f"Inference Error: {e}")
                    result_text = "Infer Error"

            if response_queue:
                response_queue.put(result_text)

        elif command == 'RESET':
            if kit: set_servos(kit, SERVO_ANGLES['default'])
            send_to_lcd(ser, "Status: Ready", "Waiting...")
            if response_queue:
                response_queue.put("Reset Complete")

        task_queue.task_done()

def camera_thread():
    global output_frame
    print("[Thread] Camera Thread started...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

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
    <title>E-Waste Sorter</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Segoe UI', sans-serif; text-align: center; background: #222; color: #fff; margin: 0; padding: 20px; }
        .container { margin-top: 20px; }
        img { border: 4px solid #555; border-radius: 8px; max-width: 100%; height: auto; box-shadow: 0 4px 8px rgba(0,0,0,0.5); }
        
        .status-box {
            background: #333; padding: 15px; margin: 20px auto; 
            max-width: 600px; border-radius: 10px; border: 1px solid #444;
        }
        h2 { margin: 0; font-size: 24px; color: #00e676; }
        
        .btn {
            background-color: #008CBA; color: white; padding: 15px 30px;
            font-size: 18px; margin: 10px; cursor: pointer; 
            border: none; border-radius: 5px; transition: 0.3s;
        }
        .btn:hover { opacity: 0.8; }
        .btn-red { background-color: #f44336; }
        .loading { color: #f1c40f; }
    </style>
</head>
<body>
    <h1>♻️ E-Waste Sorter</h1>
    
    <div>
        <img src="{{ url_for('video_feed') }}" width="640">
    </div>

    <div class="status-box">
        <div style="font-size: 14px; color: #aaa;">DETECTED OBJECT:</div>
        <h2 id="status-text">Ready to Sort</h2>
    </div>

    <div class="container">
        <button class="btn" onclick="triggerSort()">CAPTURE & SORT</button>
        <button class="btn btn-red" onclick="triggerReset()">RESET</button>
    </div>

    <script>
        function triggerSort() {
            const statusText = document.getElementById("status-text");
            statusText.innerHTML = "Scanning... ⏳";
            statusText.className = "loading";

            fetch('/api/sort', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    statusText.innerText = data.result;
                    statusText.className = ""; 
                    
                    if(data.result === "Not Found") {
                        statusText.style.color = "#f44336"; 
                    } else if (data.result.includes("Error")) {
                        statusText.style.color = "#ffa726"; 
                    } else {
                        statusText.style.color = "#00e676"; 
                    }
                })
                .catch(err => {
                    statusText.innerText = "Network Error!";
                    console.error(err);
                });
        }

        function triggerReset() {
            fetch('/api/reset', { method: 'POST' });
            const statusText = document.getElementById("status-text");
            statusText.innerText = "Ready to Sort";
            statusText.style.color = "#00e676";
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def generate_frames():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/sort', methods=['POST'])
def api_sort():
    global output_frame
    response_queue = queue.Queue()
    
    with frame_lock:
        if output_frame is not None:
            task_queue.put(('SORT', output_frame.copy(), response_queue))
        else:
            return jsonify({"result": "Error: Cam Offline"}), 500

    # Wait for the worker to return a result
    try:
        # Timeout after 5 seconds to prevent eternal hanging
        result_text = response_queue.get(timeout=5) 
    except queue.Empty:
        result_text = "Error: Timeout"
    
    return jsonify({"result": result_text})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    task_queue.put(('RESET', None, None))
    return jsonify({"status": "Reset triggered"})

if __name__ == '__main__':
    t_cam = threading.Thread(target=camera_thread, daemon=True)
    t_cam.start()

    t_inf = threading.Thread(target=inference_worker, daemon=True)
    t_inf.start()

    print("Starting Web Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
