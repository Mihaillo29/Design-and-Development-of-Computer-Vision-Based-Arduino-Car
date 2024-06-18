import serial
import time
import tkinter as tk
from PIL import Image, ImageTk

# Configure the Bluetooth serial port
ser = serial.Serial('COM5', 9600)  # Replace 'COM5' with the appropriate Bluetooth serial port

# Define key mappings
KEY_MAP = {
    'w': 'F',  # Forward
    's': 'B',  # Backward
    'a': 'L',  # Left
    'd': 'R',  # Right
    ' ': 'S',  # Stop (space bar)
}

def send_command(command):
    ser.write(command.encode())
    time.sleep(0.1)  # Wait for the command to be sent

def on_key_press(event):
    global pressed_keys
    key = event.char.lower()  # Convert to lowercase
    pressed_keys.add(key)
    
    # Check for combination key press
    if 'w' in pressed_keys:
        if 'a' in pressed_keys:
            command = 'G'  # Forward Left
        elif 'd' in pressed_keys:
            command = 'I'  # Forward Right
        else:
            command = 'F'  # Forward
    elif 's' in pressed_keys:
        if 'a' in pressed_keys:
            command = 'H'  # Backward Left
        elif 'd' in pressed_keys:
            command = 'J'  # Backward Right
        else:
            command = 'B'  # Backward
    elif key in KEY_MAP:
        command = KEY_MAP[key]
    else:
        return

    send_command(command)
    if command == 'F':
        status_label.config(text="Moving forward...")
    elif command == 'B':
        status_label.config(text="Moving backward...")
    elif command == 'L':
        status_label.config(text="Turning left...")
    elif command == 'R':
        status_label.config(text="Turning right...")
    elif command == 'S':
        status_label.config(text="Stopping...")
    elif command == 'I':
        status_label.config(text="Moving forward and right...")
    elif command == 'G':
        status_label.config(text="Moving forward and left...")
    elif command == 'H':
        status_label.config(text="Moving backward and right...")
    elif command == 'J':
        status_label.config(text="Moving backward and left...")
    elif command == 'X':
        status_label.config(text="Moving backward and right...")

def on_key_release(event):
    global pressed_keys
    key = event.char.lower()  # Convert to lowercase
    pressed_keys.discard(key)
    
    # Check if 'w' or 'd' is released, then stop the car if the other key is still pressed
    if key == 'w' and 'd' in pressed_keys:
        send_command('R')  # Keep turning right
    elif key == 'd' and 'w' in pressed_keys:
        send_command('F')  # Keep moving forward
    elif key in KEY_MAP:
        command = KEY_MAP[key]
        if command in ['F', 'B', 'L', 'R']:
            send_command('S')
            status_label.config(text="Stopped.")

def quit_program():
    send_command('S')
    status_label.config(text="Quitting...")
    ser.close()
    root.quit()

# Create GUI
root = tk.Tk()
root.title("Car Controller")

# Set initial window size
root.geometry("800x600")  # Adjust the size as needed

# Load background image and resize it to fit the window
background_image = Image.open("background.jpg")
background_image = background_image.resize((800, 600), Image.BILINEAR)  # Adjust size to match window size
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

instruction_label = tk.Label(root, text="Use WASD keys to control the car:", bg='white', font=("Arial", 16))
instruction_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

status_label = tk.Label(root, text="Press WASD keys to move the car.", bg='white', font=("Arial", 14))
status_label.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

pressed_keys = set()  # To track currently pressed keys

root.bind("<KeyPress>", on_key_press)
root.bind("<KeyRelease>", on_key_release)

quit_button = tk.Button(root, text="Quit", command=quit_program, font=("Arial", 14), bg="#C0392B", fg="white", padx=10, pady=5, bd=0)
quit_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

root.mainloop()
