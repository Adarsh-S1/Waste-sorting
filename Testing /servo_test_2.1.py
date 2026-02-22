import time
from adafruit_servokit import ServoKit

# Initialize the servo kit
kit = ServoKit(channels=16)

# --- NEW ---
# Define the channels for the four servos you want to control
servo_channels = [0, 1, 2, 3]

# Set the pulse width range for all specified MG90S servos
# This is crucial for them to move correctly
print(f"Setting pulse width for channels {servo_channels}...")
for channel in servo_channels:
    kit.servo[channel].set_pulse_width_range(500, 2500)
# --- END NEW ---


def test_servos(channels, min_angle=0, max_angle=180, step=10, delay=0.5):
    """Sweeps a list of servos back and forth in unison."""
    print(f"Testing servos on channels {channels}...")
    
    # Sweep from min to max
    for angle in range(min_angle, max_angle + 1, step):
        print(f"Set angle to {angle}")
        # Set the angle for each servo in the list
        for channel in channels:
            kit.servo[channel].angle = angle
        time.sleep(delay)
        
    # Sweep from max to min
    for angle in range(max_angle, min_angle - 1, -step):
        print(f"Set angle to {angle}")
        # Set the angle for each servo in the list
        for channel in channels:
            kit.servo[channel].angle = angle
        time.sleep(delay)
        
    print(f"Finished testing servos on channels {channels}.")


if __name__ == "__main__":
    # Call the function, passing the list of servo channels
    test_servos(channels=servo_channels)
