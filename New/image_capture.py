import cv2
import os
import time
import datetime

def main():
    # 1. Setup the output directory
    output_folder = "captured_frames_1sec"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # 2. Initialize the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Configuration
    CAPTURE_INTERVAL = 1.0  # Seconds between captures

    # State variables
    is_capturing = False
    frame_count = 0
    last_capture_time = 0  # To track when we last saved an image

    print(f"--- Interval Capture Program ({CAPTURE_INTERVAL}s) ---")
    print("Press 's' to START/STOP capturing.")
    print("Press 'q' to QUIT the program.")

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Get key press
        key = cv2.waitKey(1) & 0xFF

        # --- Toggle Logic ---
        if key == ord('s'):
            is_capturing = not is_capturing
            if is_capturing:
                print(f"\n[STARTED] Capturing every {CAPTURE_INTERVAL} second(s)...")
                # Reset timer so it captures immediately upon starting
                last_capture_time = 0 
            else:
                print("\n[STOPPED] Capturing paused.")

        # --- Quit Logic ---
        if key == ord('q'):
            print("\nExiting program...")
            break

        # --- Interval Saving Logic ---
        current_time = time.time()
        
        # Check if capturing is ON AND enough time has passed
        if is_capturing and (current_time - last_capture_time >= CAPTURE_INTERVAL):
            
            # Create timestamp for filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_folder}/img_{timestamp}.jpg"
            
            # Save the clean frame
            cv2.imwrite(filename, frame)
            
            print(f"Saved: {filename}")
            frame_count += 1
            last_capture_time = current_time  # Reset the clock

        # --- Display Logic ---
        display_frame = frame.copy()

        if is_capturing:
            # Visual Feedback
            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display_frame, f"REC ({CAPTURE_INTERVAL}s interval)", (50, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "Press 's' to Start | 'q' to Quit", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Camera Feed', display_frame)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal frames captured: {frame_count}")

if __name__ == "__main__":
    main()