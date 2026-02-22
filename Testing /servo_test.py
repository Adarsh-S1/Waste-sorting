from gpiozero import Servo

servo = Servo(17,min_pulse_width=0.5/1000, max_pulse_width=2.4/1000)
servo.value = 0.0


while True:

    num=float(input("Enter a number between -1 and 1: "))

    if num < -1 or num > 1:
        print("Invalid input. Please enter a number between -1 and 1.")
        continue
    servo.value = num
    print(f"Servo moved to position: {num}")
    input("Press Enter to continue...")

    if num == 0:
        break


