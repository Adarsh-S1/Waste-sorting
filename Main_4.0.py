import cv2
import numpy as np
import time
import serial
import threading
import queue
from flask import Flask, Response, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import logging
import json
import os
from tflite_runtime.interpreter import Interpreter
from adafruit_servokit import ServoKit

# --- LOGGING SETUP ---
logging.basicConfig(
    filename='sorter.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

# --- CONFIGURATION ---
TM_PIXEL_SCALE = 127.5   # Teachable Machine normalization: maps [0, 255] → [-1.0, 1.0]
MODEL_PATH = 'model_2.tflite'
LABEL_PATH = 'label_2.txt'

CAMERA_WIDTH = 640  # Increased for better crop quality
CAMERA_HEIGHT = 480
FRAME_RATE = 20

INPUT_WIDTH = 224
INPUT_HEIGHT = 224
CONFIDENCE_THRESHOLD = 0.85 # Adjusted slightly for averaged results

NUM_SERVOS = 4
DEFAULT_SERVO_ANGLES = {
    'Battery': [90, 90, 90, 180],
    'PCB':     [90, 90, 0, 90],
    'metal':   [0, 90, 90, 90],
    'plastic': [90, 180, 90, 90], 
    'default': [90, 90, 90, 90],
}

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'servo_config.json')

def load_servo_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Error loading servo config: {e}")
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_SERVO_ANGLES, f, indent=4)
    except Exception as e:
        log.error(f"Error writing default config: {e}")
    return DEFAULT_SERVO_ANGLES.copy()

def save_servo_config():
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(SERVO_ANGLES, f, indent=4)
    except Exception as e:
        log.error(f"Error saving servo config: {e}")

SERVO_ANGLES = load_servo_config()

SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 9600
# --- END CONFIGURATION ---

# --- GLOBAL STATE ---
task_queue = queue.Queue()

class FrameBuffer:
    """Thread-safe wrapper around the latest camera frame."""
    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def write(self, frame):
        with self._lock:
            self._frame = frame.copy()

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

frame_buffer = FrameBuffer()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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
    """Move all servos to the given angles. Emits a WebSocket alert on failure."""
    if kit is None:
        log.error("set_servos called but ServoKit is not initialized.")
        socketio.emit('hardware_error', {'msg': 'Servo not initialized!'})
        return
    try:
        for i in range(NUM_SERVOS):
            kit.servo[i].angle = angles[i]
            time.sleep(0.05)  # Allow each servo time to physically move
        log.info(f"Servos moved to: {angles}")
    except Exception as e:
        log.error(f"Servo failure: {e}")
        socketio.emit('hardware_error', {'msg': f'Servo failure: {e}'})
        raise

def send_to_lcd(ser, line1, line2=""):
    """Send two lines to the LCD over serial. Warns if text is truncated."""
    if ser is None:
        return
    try:
        if len(line1) > 16:
            log.warning(f"LCD line1 truncated: '{line1}'")
        if len(line2) > 16:
            log.warning(f"LCD line2 truncated: '{line2}'")
        message = f"{line1[:16]}|{line2[:16]}\n"
        ser.write(message.encode('utf-8'))
    except Exception as e:
        log.error(f"LCD serial error: {e}")

def capture_n_distinct_frames(buffer, n=3, timeout=2.0):
    """
    Capture n visually distinct frames from the FrameBuffer.
    Returns fewer than n frames if timeout is reached first.
    """
    frames = []
    last_frame = None
    deadline = time.time() + timeout

    while len(frames) < n and time.time() < deadline:
        frame = buffer.read()
        if frame is not None:
            if last_frame is None or not np.array_equal(frame, last_frame):
                frames.append(frame)
                last_frame = frame
        time.sleep(0.03)

    if len(frames) < n:
        log.warning(f"Only captured {len(frames)}/{n} distinct frames before timeout.")
    return frames

# --- WORKER THREADS ---

def inference_worker():
    log.info("[Thread] Inference Worker started.")
    kit = None
    ser = None
    interpreter = None
    labels = []

    try:
        kit = ServoKit(channels=16)
        log.info("ServoKit initialized.")
    except Exception as e:
        log.warning(f"ServoKit init failed: {e}")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        send_to_lcd(ser, "System Ready", "Web Connected")
        log.info("Serial/LCD initialized.")
    except Exception as e:
        log.warning(f"Serial/LCD init failed: {e}")

    try:
        labels = load_labels(LABEL_PATH)
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        log.info(f"Model loaded. Labels: {labels}")
    except Exception as e:
        log.critical(f"Model loading failed: {e}")

    if kit:
        set_servos(kit, SERVO_ANGLES['default'])

    while True:
        command, _, response_queue = task_queue.get()

        if command == 'SORT' and interpreter:
            socketio.emit('sort_progress', {'msg': '📷 Capturing frames...'})
            log.info("SORT command received. Capturing frames.")

            # Capture 3 genuinely distinct frames
            frames = capture_n_distinct_frames(frame_buffer, n=3, timeout=2.0)

            all_scores = []
            for i, frame in enumerate(frames):
                socketio.emit('sort_progress', {'msg': f'🧠 Analyzing frame {i+1}/{len(frames)}...'})

                # Pre-process: BGR → RGB, center crop, resize
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = center_crop_and_resize(rgb, (INPUT_WIDTH, INPUT_HEIGHT))
                input_data = np.expand_dims(processed, axis=0)

                # Normalize using Teachable Machine convention
                if input_details[0]['dtype'] == np.float32:
                    input_data = (input_data.astype(np.float32) / TM_PIXEL_SCALE) - 1.0

                # Run inference
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                all_scores.append(
                    interpreter.get_tensor(output_details[0]['index'])[0]
                )

            if all_scores:
                avg_scores = np.mean(all_scores, axis=0)
                class_id = int(np.argmax(avg_scores))
                confidence = float(avg_scores[class_id])
                class_name = labels[class_id]
                log.info(f"Result: {class_name} ({confidence:.0%})")

                if confidence > CONFIDENCE_THRESHOLD:
                    result_text = f"{class_name} ({confidence:.0%})"
                    send_to_lcd(ser, f"Found: {class_name}", f"Conf:{confidence:.0%}")
                    socketio.emit('sort_result', {
                        'result': result_text,
                        'class': class_name,
                        'confidence': confidence
                    })

                    if class_name in SERVO_ANGLES:
                        set_servos(kit, SERVO_ANGLES[class_name])
                        time.sleep(1.5)           # Wait for object to physically drop/slide
                        set_servos(kit, SERVO_ANGLES['default'])   # Auto-reset
                        send_to_lcd(ser, "Ready", "Next object?")
                        log.info("Servos auto-reset to default after sort.")
                else:
                    result_text = "Low Confidence — Try Again"
                    send_to_lcd(ser, "Low Confidence", "Try Again")
                    socketio.emit('sort_result', {'result': result_text, 'class': None, 'confidence': confidence})
                    log.warning(f"Low confidence: {class_name} at {confidence:.0%}")
            else:
                result_text = "Camera Error"
                socketio.emit('sort_result', {'result': result_text, 'class': None, 'confidence': 0})
                log.error("No frames captured for inference.")

            if response_queue:
                response_queue.put(result_text)

        elif command == 'RESET':
            if kit:
                set_servos(kit, SERVO_ANGLES['default'])
            send_to_lcd(ser, "Status: Ready", "Waiting...")
            socketio.emit('sort_result', {'result': 'System Reset ✅', 'class': None, 'confidence': 0})
            log.info("Manual reset triggered.")
            if response_queue:
                response_queue.put("Reset Complete")

        task_queue.task_done()

def camera_thread():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    log.info("[Thread] Camera thread started.")
    while True:
        ret, frame = cap.read()
        if ret:
            frame_buffer.write(frame)
        time.sleep(0.01)

# --- FLASK APPLICATION ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>E-Waste Sorter Pro</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', sans-serif; text-align: center;
               background: #1a1a1a; color: #fff; margin: 0; padding: 20px; }
        .container { margin-top: 20px; }
        img { border: 4px solid #333; border-radius: 12px; max-width: 100%; height: auto; }
        .status-box { background: #262626; padding: 20px; margin: 20px auto;
                      max-width: 500px; border-radius: 10px; border-left: 5px solid #00e676; }
        h2 { margin: 5px 0; color: #00e676; }
        .btn { background: #008CBA; color: #fff; padding: 15px 40px; font-size: 18px;
               margin: 10px; cursor: pointer; border: none; border-radius: 8px; font-weight: bold; }
        .btn-red { background: #d32f2f; }
        .loading { color: #ffeb3b; }
        .alert-box { display: none; background: #b71c1c; color: #fff;
                     padding: 10px 20px; border-radius: 8px; margin: 10px auto;
                     max-width: 500px; font-weight: bold; }
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0;
                 width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.8); }
        .modal-content { background-color: #1a1a1a; margin: 5% auto; padding: 20px;
                         border: 1px solid #333; width: 90%; max-width: 600px; border-radius: 12px; color: #fff; }
        .modal-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 20px;}
        .modal-header h2 { margin: 0; color: #00e676; }
        .modal-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        .modal-table th, .modal-table td { border: 1px solid #333; padding: 10px; text-align: center; }
        .modal-table input { width: 60px; padding: 5px; background: #262626; color: #fff; border: 1px solid #555; border-radius: 4px; text-align: center; }
        .close-btn { color: #aaa; font-size: 28px; font-weight: bold; cursor: pointer; }
        .close-btn:hover { color: #fff; }
    </style>
</head>
<body>
    <h1>♻️ E-Waste Smart Sorter</h1>
    <div><img src="{{ url_for('video_feed') }}"></div>

    <div id="alert-box" class="alert-box"></div>

    <div class="status-box">
        <small style="color: #888;">AI CLASSIFICATION</small>
        <h2 id="status-text">System Ready</h2>
    </div>
    <div class="container">
        <button class="btn" onclick="triggerSort()">SCAN OBJECT</button>
        <button class="btn btn-red" onclick="triggerReset()">RESET</button>
        <button class="btn" style="background: #555;" onclick="openConfigModal()">⚙️ SERVO CONFIG</button>
    </div>

    <div id="configModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>Servo Configuration</h2>
                <span class="close-btn" onclick="closeConfigModal()">&times;</span>
            </div>
            <table class="modal-table" id="configTable">
                <thead>
                    <tr>
                        <th>Waste Type</th>
                        <th>Servo 1</th>
                        <th>Servo 2</th>
                        <th>Servo 3</th>
                        <th>Servo 4</th>
                    </tr>
                </thead>
                <tbody id="configTableBody">
                </tbody>
            </table>
            <div>
                <button class="btn" onclick="saveConfig()">SAVE</button>
                <button class="btn btn-red" onclick="closeConfigModal()">CANCEL</button>
            </div>
        </div>
    </div>

    <script>
        const socket = io();   // One persistent connection — stays open

        // Live progress during inference (e.g. "Capturing frame 1/3...")
        socket.on('sort_progress', (data) => {
            const st = document.getElementById('status-text');
            st.innerText = data.msg;
            st.className = 'loading';
            st.style.color = '#ffeb3b';
        });

        // Final classification result
        socket.on('sort_result', (data) => {
            const st = document.getElementById('status-text');
            st.innerText = data.result;
            st.className = '';
            st.style.color = (data.class !== null) ? '#00e676' : '#ff5252';
        });

        // Hardware error pushed by the server (e.g. servo failure)
        socket.on('hardware_error', (data) => {
            const box = document.getElementById('alert-box');
            box.innerText = '⚠️ Hardware Alert: ' + data.msg;
            box.style.display = 'block';
            setTimeout(() => { box.style.display = 'none'; }, 6000);
        });

        function triggerSort() {
            document.getElementById('status-text').innerText = 'Starting... 🔍';
            document.getElementById('status-text').style.color = '#ffeb3b';
            socket.emit('start_sort');   // Fire and forget — results arrive via events
        }

        function triggerReset() {
            socket.emit('reset');
            document.getElementById('status-text').innerText = 'System Ready';
            document.getElementById('status-text').style.color = '#00e676';
        }

        function openConfigModal() {
            fetch('/api/servo_config')
                .then(r => r.json())
                .then(data => {
                    const tbody = document.getElementById('configTableBody');
                    tbody.innerHTML = '';
                    for (const [key, angles] of Object.entries(data)) {
                        let html = `<tr><td>${key}</td>`;
                        for (let i = 0; i < 4; i++) {
                            html += `<td><input type="number" min="0" max="180" step="1" value="${angles[i]}"></td>`;
                        }
                        html += `</tr>`;
                        tbody.innerHTML += html;
                    }
                    document.getElementById('configModal').style.display = 'block';
                });
        }

        function closeConfigModal() {
            document.getElementById('configModal').style.display = 'none';
        }

        function saveConfig() {
            const rows = document.getElementById('configTableBody').getElementsByTagName('tr');
            const newConfig = {};
            for (let row of rows) {
                const cells = row.getElementsByTagName('td');
                const key = cells[0].innerText;
                const angles = [];
                for (let i = 1; i <= 4; i++) {
                    const input = cells[i].getElementsByTagName('input')[0];
                    angles.push(parseInt(input.value));
                }
                newConfig[key] = angles;
            }

            fetch('/api/servo_config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newConfig)
            })
            .then(r => r.json().then(data => ({status: r.status, body: data})))
            .then(res => {
                if (res.status === 200) {
                    alert('Servo configuration saved successfully!');
                    closeConfigModal();
                } else {
                    alert('Error: ' + res.body.error);
                }
            })
            .catch(err => alert('Request failed: ' + err));
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

def generate_frames():
    while True:
        frame = frame_buffer.read()
        if frame is None:
            continue
        flag, encoded = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + bytearray(encoded)
            + b'\r\n'
        )

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/servo_config', methods=['GET', 'POST'])
def api_servo_config():
    global SERVO_ANGLES
    if request.method == 'GET':
        return jsonify(SERVO_ANGLES)
    elif request.method == 'POST':
        data = request.json
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON format"}), 400
        
        for k, v in data.items():
            if not isinstance(v, list) or len(v) != 4:
                return jsonify({"error": f"Value for '{k}' must be a list of 4 integers"}), 400
            for val in v:
                if not isinstance(val, int) or val < 0 or val > 180:
                    return jsonify({"error": f"Values must be integers between 0 and 180 (got {val})"}), 400
        
        SERVO_ANGLES.update(data)
        save_servo_config()
        return jsonify({"status": "success"})

@socketio.on('start_sort')
def handle_sort():
    """WebSocket event: client requests a sort. Non-blocking."""
    log.info("WebSocket 'start_sort' received.")
    task_queue.put(('SORT', None, None))

@socketio.on('reset')
def handle_reset():
    """WebSocket event: client requests a reset."""
    log.info("WebSocket 'reset' received.")
    task_queue.put(('RESET', None, None))

if __name__ == '__main__':
    log.info("Starting E-Waste Sorter application.")
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=inference_worker, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False,allow_unsafe_werkzeug=True)