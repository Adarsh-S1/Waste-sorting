# E-Waste Sorter — Code Modification Guide

This document contains all suggested improvements to be applied to the existing
`Main_4.0.py` code. Apply each section in order. Every change is explained with
the exact reason so you understand what problem it solves.

---

## 0. New Dependencies to Add


Add this import block at the top of the file alongside the existing imports:

```python
# REPLACE this line:
from flask import Flask, Response, render_template_string, jsonify

# WITH this:
from flask import Flask, Response, render_template_string, jsonify
from flask_socketio import SocketIO, emit
import logging
```

---

## 1. Add Logging Setup

**Why:** Currently all errors use `print()`. If something breaks at runtime, there
is no record. A log file captures every event with timestamps.

Add this block immediately after the import section, before any other code:

```python
# --- LOGGING SETUP ---
logging.basicConfig(
    filename='sorter.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)
```

---

## 2. Add a Named Constant for Teachable Machine Normalization

**Why:** The magic number `127.5` in the inference code is not self-explanatory.
Name it clearly so anyone reading the code understands what normalization is
being applied and why.

```python
# ADD to the --- CONFIGURATION --- section:
TM_PIXEL_SCALE = 127.5   # Teachable Machine normalization: maps [0, 255] → [-1.0, 1.0]
```

---

## 3. Replace `global output_frame` with a Thread-Safe FrameBuffer Class

**Why:** The global `output_frame` variable is accessed by two threads without
consistent locking. Between the `if output_frame is not None` check and
the `.copy()` call, another thread can set it to `None`, causing a crash.
This class makes all access safe.

```python
# REMOVE these lines from --- GLOBAL STATE ---:
output_frame = None
frame_lock = threading.Lock()

# ADD this class in their place:
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
```

---

## 4. Replace `set_servos` with a Safer Version

**Why:** The current version silently swallows servo errors with just a `print`.
For a physical machine, a failed servo means an object is misrouted. The UI
must be notified and the error must be logged.

Also adds a small per-servo delay so servos have time to physically move
before the next command is issued.

```python
# REPLACE the entire set_servos function with:
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
```

---

## 5. Replace `send_to_lcd` with a Version That Logs Truncation

**Why:** The LCD silently cuts off text longer than 16 characters. This hides
bugs where you accidentally send a long string and wonder why it displays wrong.

```python
# REPLACE the entire send_to_lcd function with:
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
```

---

## 6. Add a Distinct-Frame Capture Helper

**Why:** The current 3-frame averaging loop sleeps for 50ms between grabs, which
matches the 20 FPS frame rate. This means it may grab the same frame multiple
times, making the averaging pointless. This helper only adds a frame to the
list when it is genuinely different from the last one captured.

```python
# ADD this new function after send_to_lcd:
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
```

---

## 7. Rewrite `inference_worker` with WebSocket + All Fixes Applied

**Why:** This incorporates every improvement in one place:
- Uses `frame_buffer` instead of `global output_frame`
- Uses `capture_n_distinct_frames` for real multi-frame averaging
- Emits live WebSocket progress updates to the UI
- Uses the named `TM_PIXEL_SCALE` constant
- Auto-resets servos after sorting so the machine is always ready for the next object
- Logs all results

```python
# REPLACE the entire inference_worker function with:
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
```

---

## 8. Update `camera_thread` to Use FrameBuffer

**Why:** The old thread wrote directly to the global variable. Now it uses the
thread-safe `FrameBuffer.write()` method.

```python
# REPLACE the entire camera_thread function with:
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
```

---

## 9. Update `generate_frames` to Use FrameBuffer

**Why:** The MJPEG stream generator still referenced the old global. Update it
to use `frame_buffer.read()`.

```python
# REPLACE the entire generate_frames function with:
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
```

---

## 10. Replace Flask App Initialization with SocketIO

**Why:** Flask alone cannot push events to the browser. Flask-SocketIO wraps
Flask and adds WebSocket support with minimal changes.

```python
# REPLACE this line:
app = Flask(__name__)

# WITH these two lines:
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
```

---

## 11. Replace the HTML_TEMPLATE with WebSocket-Enabled Version

**Why:** The old frontend used `fetch('/api/sort')` which blocked for up to 6
seconds with no feedback. The new frontend connects via WebSocket and receives
live progress updates (`sort_progress`) and the final result (`sort_result`).
It also listens for `hardware_error` alerts from the servo system.

```python
# REPLACE the entire HTML_TEMPLATE string with:
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
    </script>
</body>
</html>
"""
```

---

## 12. Replace HTTP API Routes with WebSocket Event Handlers

**Why:** The `/api/sort` HTTP route blocked for up to 6 seconds. The new
WebSocket event handlers are non-blocking — they drop a task on the queue
and return immediately. Results are pushed back via `socketio.emit` inside
`inference_worker`.

```python
# REMOVE these two routes entirely:
# @app.route('/api/sort', methods=['POST'])
# def api_sort(): ...
#
# @app.route('/api/reset', methods=['POST'])
# def api_reset(): ...

# REPLACE them with these WebSocket event handlers:
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
```

> **Keep** the `@app.route('/')` and `@app.route('/video_feed')` routes —
> they are unchanged.

---

## 13. Update `__main__` to Use SocketIO

**Why:** `app.run()` does not support WebSockets. Replace it with
`socketio.run()` which handles both HTTP and WebSocket connections.

```python
# REPLACE the __main__ block with:
if __name__ == '__main__':
    log.info("Starting E-Waste Sorter application.")
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=inference_worker, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
```

---

## Summary of All Changes

| # | Change | Problem Solved |
|---|--------|---------------|
| 1 | Logging setup | Silent failures, no audit trail |
| 2 | Named `TM_PIXEL_SCALE` constant | Magic number `127.5` was unreadable |
| 3 | `FrameBuffer` class | Race condition on global `output_frame` |
| 4 | Safer `set_servos` | Servo errors swallowed silently |
| 5 | LCD truncation warning | Silent data loss on display |
| 6 | `capture_n_distinct_frames` | 3-frame average was grabbing same frame |
| 7 | Rewritten `inference_worker` | Combines all fixes + WebSocket events + auto-reset |
| 8 | `camera_thread` uses FrameBuffer | Consistency with new buffer system |
| 9 | `generate_frames` uses FrameBuffer | Consistency with new buffer system |
| 10 | `SocketIO` wraps Flask | Enables WebSocket support |
| 11 | New HTML template | Live UI feedback, hardware error alerts |
| 12 | WebSocket event handlers replace HTTP routes | Non-blocking sort, server-push results |
| 13 | `socketio.run` replaces `app.run` | Required to serve WebSocket connections |

---

*Apply changes in order. Each section is self-contained with the exact old code
to remove and new code to insert.*
