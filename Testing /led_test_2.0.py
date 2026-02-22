import time
from smbus2 import SMBus

# --- Configuration ---
# IMPORTANT: Change this to the address you found with 'i2cdetect -y 1'
I2C_ADDR = 0x27 
I2C_BUS = 1 # Raspberry Pi 5 uses I2C bus 1
LCD_WIDTH = 16 # Characters per line

# --- LCD Commands ---
LCD_CHR = 1 # Mode - Sending data
LCD_CMD = 0 # Mode - Sending command

LINE_ADDRESS = [0x80, 0xC0, 0x90, 0xD0] # LCD RAM addresses for each line
LCD_BACKLIGHT = 0x08 # On

ENABLE = 0b00000100 # Enable bit

# --- Core Functions (No changes here) ---
def lcd_init():
    """Initializes the LCD."""
    lcd_byte(0x33, LCD_CMD)
    lcd_byte(0x32, LCD_CMD)
    lcd_byte(0x06, LCD_CMD)
    lcd_byte(0x0C, LCD_CMD)
    lcd_byte(0x28, LCD_CMD)
    lcd_byte(0x01, LCD_CMD)
    time.sleep(0.0005)

def lcd_byte(bits, mode):
    """Sends a byte to the data pins."""
    bits_high = mode | (bits & 0xF0) | LCD_BACKLIGHT
    bits_low = mode | ((bits << 4) & 0xF0) | LCD_BACKLIGHT
    bus.write_byte(I2C_ADDR, bits_high)
    lcd_toggle_enable(bits_high)
    bus.write_byte(I2C_ADDR, bits_low)
    lcd_toggle_enable(bits_low)

def lcd_toggle_enable(bits):
    """Toggles the enable pin."""
    time.sleep(0.0005)
    bus.write_byte(I2C_ADDR, (bits | ENABLE))
    time.sleep(0.0005)
    bus.write_byte(I2C_ADDR, (bits & ~ENABLE))
    time.sleep(0.0005)

def lcd_string(message, line):
    """Sends a string to the specified line."""
    # Truncate or pad message to fit LCD width
    message = message.ljust(LCD_WIDTH, " ")[:LCD_WIDTH]
    lcd_byte(LINE_ADDRESS[line-1], LCD_CMD)
    for i in range(LCD_WIDTH):
        lcd_byte(ord(message[i]), LCD_CHR)

def clear_lcd():
    """Clears the entire LCD display."""
    lcd_byte(0x01, LCD_CMD)
    time.sleep(0.005)

# --- Main Program Logic (UPDATED SECTION) ---
def main():
    """Main program logic with user input."""
    global bus
    bus = SMBus(I2C_BUS)

    # Initialize display
    lcd_init()
    print("LCD ready. Type text for each line.")
    print("Press CTRL+C to exit.")

    try:
        while True:
            # Clear the screen for new input
            clear_lcd()
            
            # Get input from the user for each of the four lines
            line1_text = input("Enter text for Line 1 > ")
            lcd_string(line1_text, 1)

            line2_text = input("Enter text for Line 2 > ")
            lcd_string(line2_text, 2)
            
            line3_text = input("Enter text for Line 3 > ")
            lcd_string(line3_text, 3)

            line4_text = input("Enter text for Line 4 > ")
            lcd_string(line4_text, 4)
            
            # Wait for user to decide to enter new text or exit
            input("\nDisplay updated. Press Enter to write new text or Ctrl+C to exit...")


    except KeyboardInterrupt:
        print("\nCleaning up...")
        clear_lcd()
        lcd_string("Goodbye!", 1)


if __name__ == '__main__':
    main()