import time

from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)
kit.servo[0].set_pulse_width_range(500, 2400)

def test_servo(channel, min_angle=0, max_angle=180, step=10, delay=0.1):
    print(f"Testing servo on channel {channel}...")
    for angle in range(min_angle, max_angle + 1, step):
        kit.servo[channel].angle = angle
        print(f"Set angle to {angle}")
        time.sleep(delay)
    for angle in range(max_angle, min_angle - 1, -step):
        kit.servo[channel].angle = angle
        print(f"Set angle to {angle}")
        time.sleep(delay)
    print(f"Finished testing servo on channel {channel}.")

if __name__ == "__main__":
    test_servo(channel=0)
