import serial
import time
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

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


while True:
    #send_to_lcd(ser, "Hello", "World")
    time.sleep(3)

    text = input("Enter first :")
    text2 = input("Enter Second :")
    send_to_lcd(ser, text,text2)
    time.sleep(3)
    print("send")


ser.close()
