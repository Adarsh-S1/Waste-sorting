import subprocess
import time
import sys

TARGET_SSID = "galaxy"

def check_wifi():
    try:
        # Check current connection using nmcli
        result = subprocess.check_output(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"])
        return f"yes:{TARGET_SSID}" in result.decode("utf-8")
    except:
        return False

# Loop until connected
while not check_wifi():
    print(f"Waiting for {TARGET_SSID}...")
    time.sleep(5)

print("Connected! Starting Flask...")
sys.exit(0)
