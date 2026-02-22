import cv2
import time
import serial
from adafruit_servokit import ServoKit

# --- CONFIGURATION (Derived from your file) ---
#
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FRAME_RATE = 20

NUM_SERVOS = 4
# Exact angles from your reference code
SERVO_ANGLES = {
    'Battery': [90, 90, 90, 0],
    'PCB':     [90, 90, 0, 90],
    'metal':   [180, 90, 90, 90],
    'plastic': [90, 0, 90, 90], 
    'default': [90, 90, 90, 90],
}

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600
# --- END OF CONFIGURATION ---

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
    print("Initializing Manual Sorter... 🎮")
    
    # 1. Initialization
    kit = None
    ser = None
    
    try:
        kit = ServoKit(channels=16)
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Allow Arduino to reset
        send_to_lcd(ser, "Manual Mode", "Ready to Sort")
        print("Hardware initialized.")
    except Exception as e:
        print(f"Error initializing hardware: {e}")
        # We continue even if hardware fails so you can test the keys,
        # but in production, you might want to return here.

    # Init Camera (Needed to capture keyboard events via cv2 window)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    # Set initial state
    if kit:
        set_servos(kit, SERVO_ANGLES['default'])
    
    print("\n--- MANUAL CONTROLS ---")
    print("Press '1' -> Battery")
    print("Press '2' -> PCB")
    print("Press '3' -> Metal")
    print("Press '4' -> Plastic")
    print("Press 'r' -> Reset Positions")
    print("Press 'q' -> Quit")
    print("-----------------------")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display text on screen for easy reference
            cv2.putText(frame, "1:Bat 2:PCB 3:Met 4:Plas", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Manual Sorter Feed', frame)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            # --- KEYBOARD MAPPING ---
            
            # Press '1' for Battery
            if key == ord('1'):
                print("\n[Manual: Battery]")
                send_to_lcd(ser, "Sorting:", "Battery")
                if kit: set_servos(kit, SERVO_ANGLES['Battery'])

            # Press '2' for PCB
            elif key == ord('2'):
                print("\n[Manual: PCB]")
                send_to_lcd(ser, "Sorting:", "PCB")
                if kit: set_servos(kit, SERVO_ANGLES['PCB'])

            # Press '3' for Metal
            elif key == ord('3'):
                print("\n[Manual: Metal]")
                send_to_lcd(ser, "Sorting:", "Metal")
                if kit: set_servos(kit, SERVO_ANGLES['metal'])

            # Press '4' for Plastic
            elif key == ord('4'):
                print("\n[Manual: Plastic]")
                send_to_lcd(ser, "Sorting:", "Plastic")
                if kit: set_servos(kit, SERVO_ANGLES['plastic'])

            # Press 'r' for Reset
            elif key == ord('r'):
                print("\n[Resetting]")
                send_to_lcd(ser, "Manual Mode", "Ready")
                if kit: set_servos(kit, SERVO_ANGLES['default'])

            # Press 'q' for Quit
            elif key == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass 

    finally:
        print("\nShutting down...")
        cap.release()
        cv2.destroyAllWindows()
        if kit:
            set_servos(kit, SERVO_ANGLES['default'])
        if ser:
            send_to_lcd(ser, "System Offline.")
            ser.close()
        print("Done.")

if __name__ == '__main__':
    main()
