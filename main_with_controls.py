import cv2
import numpy as np
import time
import serial
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
# --- END OF CONFIGURATION ---

def load_labels(path):
    """Loads labels from a text file."""
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def set_servos(kit, angles):
    """Sets the angles for all servos."""
    try:
        for i in range(NUM_SERVOS):
            kit.servo[i].angle = angles[i]
        print(f"Moving servos to: {angles}")
    except Exception as e:
        print(f"Error moving servos: {e}")

def send_to_lcd(ser, line1, line2=""):
    """
    Sends two lines of text to the Arduino via Serial.
    The lines are separated by '|' and terminated by '\n'.
    The Arduino code handles the splitting and display.
    """
    # Truncate lines to 16 characters to prevent buffer overflow
    line1_trunc = line1[:16]
    line2_trunc = line2[:16]
    
    # Format message: Line1|Line2\n
    message = f"{line1_trunc}|{line2_trunc}\n"
    
    # Encode and send the message
    ser.write(message.encode('utf-8'))
    print(f"Serial Sent to LCD: '{line1_trunc}' / '{line2_trunc}'")

def main():
    """Main function to run the e-waste sorter."""
    print("Initializing e-waste sorter... 🤖")
    
    # 1. Initialization
    try:
        kit = ServoKit(channels=16)
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Allow Arduino to reset
        send_to_lcd(ser, "E-Waste Sorter", "Starting...")
        print("Devices initialized.")
    except Exception as e:
        print(f"Error initializing devices: {e}")
        return

    # Load Model
    try:
        labels = load_labels(LABEL_PATH)
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        send_to_lcd(ser, "Model Error!")
        return

    # Init Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        send_to_lcd(ser, "Camera Error!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
    
    # Set initial state
    set_servos(kit, SERVO_ANGLES['default'])
    send_to_lcd(ser, "Status: Ready", "Press 'c' to Sort")
    print("\n--- CONTROLS ---")
    print("Press 'c' to Capture & Sort")
    print("Press 'r' to Reset/Ready")
    print("Press 'q' to Quit")
    print("----------------")

    # 2. Main Loop
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Always show the live feed so user can position object
            cv2.imshow('E-Waste Sorter Feed', frame)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            # --- TRIGGER: Capture & Classify ('c') ---
            if key == ord('c'):
                print("\n[Capturing Frame...]")
                
                # Pre-process the CURRENT frame
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
                        # Move servos and HOLD position
                        set_servos(kit, SERVO_ANGLES[class_name])
                    else:
                        print(f"Warning: No servo angle for {class_name}")
                else:
                    print(f"Low confidence: {confidence:.2f}")
                    send_to_lcd(ser, "Unknown Object", "Try Again")

            # --- TRIGGER: Reset ('r') ---
            elif key == ord('r'):
                print("\n[Resetting System...]")
                set_servos(kit, SERVO_ANGLES['default'])
                send_to_lcd(ser, "Status: Ready", "Press 'c' to Sort")

            # --- TRIGGER: Quit ('q') ---
            elif key == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass 

    finally:
        # 4. Cleanup
        print("\nShutting down...")
        cap.release()
        cv2.destroyAllWindows()
        set_servos(kit, SERVO_ANGLES['default'])
        send_to_lcd(ser, "System Offline.")
        ser.close()
        print("Done.")

if __name__ == '__main__':
    main()
