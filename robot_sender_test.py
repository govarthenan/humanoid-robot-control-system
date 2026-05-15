import socket
import time

PI_IP = "100.99.129.111"
PORT = 5005

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((PI_IP, PORT))

print("Connected to Raspberry Pi")

angles = [90, 100, 80, 110, 90]

try:
    for angle in angles:
        msg = f"HEAD_YAW={angle}\n"
        client.sendall(msg.encode())
        print("Sent:", msg.strip())
        time.sleep(2)

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    client.close()