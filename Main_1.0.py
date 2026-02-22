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

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600
# --- END CONFIGURATION ---

# --- GLOBAL STATE ---
# Queue to send commands to the inference thread
# Items in queue will be tuples: (command_string, frame_data)
task_queue = queue.Queue()

# Global variable to store the latest frame for the video feed
output_frame = None
frame_lock = threading.Lock()

# Flask App
app = Flask(__name__)

# --- HARDWARE FUNCTIONS ---

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def set_servos(kit, angles):
    try:
        for i in range(NUM_SERVOS):
            kit.servo[i].angle = angles[i]
        print(f"Moving servos to: {angles}")
    except Exception as e:
        print(f"Error moving servos: {e}")

def send_to_lcd(ser, line1, line2=""):
    try:
        line1_trunc = line1[:16]
        line2_trunc = line2[:16]
        message = f"{line1_trunc}|{line2_trunc}\n"
        ser.write(message.encode('utf-8'))
        print(f"Serial Sent: '{line1_trunc}' / '{line2_trunc}'")
    except Exception as e:
        print(f"LCD Error: {e}")

# --- WORKER THREADS ---

def inference_worker():
    """
    Runs in a separate thread.
    Waits for commands from the task_queue and controls hardware.
    """
    print("[Thread] Inference Worker started...")
    
    # Initialize Hardware INSIDE or used only by this thread to prevent conflicts
    try:
        kit = ServoKit(channels=16)
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Arduino reset
        send_to_lcd(ser, "System Ready", "Web Connected")
        
        # Load Model
        labels = load_labels(LABEL_PATH)
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("[Thread] Model & Hardware Ready.")
    except Exception as e:
        print(f"[Thread] Hardware/Model Init Error: {e}")
        return

    # Set Initial State
    set_servos(kit, SERVO_ANGLES['default'])

    while True:
        # Block until a task is received
        command, frame = task_queue.get()
        
        if command == 'SORT':
            if frame is None: 
                print("Error: No frame provided for sort.")
                task_queue.task_done()
                continue
            
            print("Processing image...")
            # Pre-process
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (INPUT_WIDTH, INPUT_HEIGHT))
            input_data = np.expand_dims(image_resized, axis=0).astype(input_details[0]['dtype'])

            # Inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Result
            scores = output_data[0]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            class_name = labels[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                print(f"Detected: {class_name} ({confidence:.1%})")
                send_to_lcd(ser, f"Found: {class_name}", f"Conf:{confidence:.0%}")
                if class_name in SERVO_ANGLES:
                    set_servos(kit, SERVO_ANGLES[class_name])
                else:
                    print(f"Warning: No servo angle for {class_name}")
            else:
                print(f"Low confidence: {confidence:.2f}")
                send_to_lcd(ser, "Unknown Object", "Try Again")

        elif command == 'RESET':
            print("Resetting hardware...")
            set_servos(kit, SERVO_ANGLES['default'])
            send_to_lcd(ser, "Status: Ready", "Waiting...")

        task_queue.task_done()

def camera_thread():
    """
    Runs in a separate thread.
    Continuously captures frames from the camera.
    """
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
            # Update the global frame safely
            with frame_lock:
                output_frame = frame.copy()
        time.sleep(0.01) # Small sleep to prevent CPU hogging

# --- FLASK APPLICATION ---

# Simple HTML Interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>E-Waste Sorter Control</title>
    <style>
        body { font-family: sans-serif; text-align: center; background: #f0f0f0; }
        h1 { color: #333; }
        .container { margin-top: 20px; }
        img { border: 5px solid #444; border-radius: 5px; max-width: 100%; }
        .btn {
            background-color: #008CBA; color: white; padding: 15px 32px;
            text-align: center; display: inline-block; font-size: 16px;
            margin: 10px 2px; cursor: pointer; border: none; border-radius: 4px;
        }
        .btn-red { background-color: #f44336; }
    </style>
</head>
<body>
    <h1>E-Waste Sorter 🤖</h1>
    <div>
        <img src="{{ url_for('video_feed') }}" width="640">
    </div>
    <div class="container">
        <button class="btn" onclick="triggerAction('sort')">CAPTURE & SORT</button>
        <button class="btn btn-red" onclick="triggerAction('reset')">RESET</button>
    </div>
    <script>
        function triggerAction(action) {
            fetch('/api/' + action, { method: 'POST' })
                .then(response => response.json())
                .then(data => console.log(data));
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def generate_frames():
    """Generator function for Flask video streaming."""
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            # Encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        
        # Yield the output frame in byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/sort', methods=['POST'])
def api_sort():
    global output_frame
    with frame_lock:
        if output_frame is not None:
            # Send a COPY of the current frame to the worker thread
            task_queue.put(('SORT', output_frame.copy()))
            return jsonify({"status": "Sorting triggered"})
        else:
            return jsonify({"status": "Error: No frame available"}), 500

@app.route('/api/reset', methods=['POST'])
def api_reset():
    task_queue.put(('RESET', None))
    return jsonify({"status": "Reset triggered"})

# --- MAIN ---
if __name__ == '__main__':
    # 1. Start Camera Thread
    t_cam = threading.Thread(target=camera_thread, daemon=True)
    t_cam.start()

    # 2. Start Inference/Hardware Worker Thread
    t_inf = threading.Thread(target=inference_worker, daemon=True)
    t_inf.start()

    # 3. Start Flask Web Server
    # host='0.0.0.0' makes it accessible from other devices on the network
    print("Starting Web Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)