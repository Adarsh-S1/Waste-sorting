import cv2
import numpy as np
import time
import serial
import sys
import os
from tflite_runtime.interpreter import Interpreter
from adafruit_servokit import ServoKit

# --- CONFIGURATION ---
# Adjust these values for your specific setup

# Input Image Path (Replace with your actual image filename)
IMAGE_FILENAME = 'img/img_20251211_154401.jpg' 

# Model and Label Paths
MODEL_PATH = 'model_2.tflite'
LABEL_PATH = 'label_2.txt'

# TFLite Model Settings
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
CONFIDENCE_THRESHOLD = 0.70  # Only act if confidence is over 70%

# Servo Configuration
NUM_SERVOS = 4
# Map your model's class names to the angles for each of the 4 servos.
SERVO_ANGLES = {
    'Battery': [90, 90, 90, 180],
    'PCB':     [90, 90, 0, 90],
    'metal':   [0, 90, 90, 90],
    'plastic': [90, 180, 90, 90], 
    'default': [90, 90, 90, 90],
}

# Serial Configuration for Arduino
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
    """Main function to run the single-image e-waste sorter."""
    print("Initializing Image Sorter... 🤖")
    
    # 1. Initialization
    try:
        kit = ServoKit(channels=16)
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(4) 
        send_to_lcd(ser, "Sorter Ready", "Load Image...")
        print("Hardware initialized.")
    except Exception as e:
        print(f"Error initializing devices: {e}")
        return

    # Load TFLite model
    try:
        labels = load_labels(LABEL_PATH)
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        print(f"Error loading model: {e}")
        send_to_lcd(ser, "Model Error!")
        return

    # 2. Load and Pre-process Image
    if not os.path.exists(IMAGE_FILENAME):
        print(f"Error: Image file '{IMAGE_FILENAME}' not found.")
        send_to_lcd(ser, "File Not Found")
        return

    print(f"Loading image: {IMAGE_FILENAME}")
    image = cv2.imread(IMAGE_FILENAME)
    
    if image is None:
        print("Error: Could not decode image.")
        return

    # Resize and prepare for model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (INPUT_WIDTH, INPUT_HEIGHT))
    input_data = np.expand_dims(image_resized, axis=0).astype(input_details[0]['dtype'])

    # 3. Perform Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    scores = output_data[0]
    class_id = np.argmax(scores)
    confidence = scores[class_id]
    class_name = labels[class_id]

    # Show result
    print(f"\n--- RESULT ---")
    print(f"Detected: {class_name}")
    print(f"Confidence: {confidence:.2%}")
    print(f"--------------\n")

    # 4. Decision Logic and Action
    if confidence > CONFIDENCE_THRESHOLD:
        # Update LCD
        send_to_lcd(ser, f"Class: {class_name}", f"Conf: {confidence:.0%}")
        
        # Move servos and MAINTAIN position
        if class_name in SERVO_ANGLES:
            target_angles = SERVO_ANGLES[class_name]
            set_servos(kit, target_angles)
            print(">> Position maintained. Servos holding target angles.")
        else:
            print(f"Warning: Class '{class_name}' not defined in angles.")
    else:
        print("Confidence too low. No action taken.")
        send_to_lcd(ser, "Low Confidence", f"Max: {confidence:.0%}")
        # Optionally move to default if confidence is low
        set_servos(kit, SERVO_ANGLES['default'])

    # 5. Hold Program to Maintain State
    try:
        # We need to keep the script running so the Servos don't lose signal/power 
        # (depending on the specific HAT/driver behavior)
        input("\nPress [Enter] to cleanup and exit...")
    except KeyboardInterrupt:
        pass

    # 6. Cleanup
    print("Shutting down...")
    # Optional: Reset to default on exit? 
    # Uncomment the next line if you want them to reset when you close the script
    # set_servos(kit, SERVO_ANGLES['default']) 
    
    send_to_lcd(ser, "System Offline")
    ser.close()
    print("Goodbye!")

if __name__ == '__main__':
    main()
