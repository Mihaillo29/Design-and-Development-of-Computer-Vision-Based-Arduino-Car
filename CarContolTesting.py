####################################################### U Can Control Ur ARduino Car With this Code Though U needed the Arduino Control Programming Acc to this which is easy #############################################
################# Find the code in Arduino Code Section ##########################


import serial
import time
import keyboard

# Configure the Bluetooth serial port
ser = serial.Serial('COM5', 9600)  # Replace 'COM3' with the appropriate Bluetooth serial port

def send_command(command):
    ser.write(command.encode())
    time.sleep(0.1)  # Wait for the command to be sent

while True:
    # Prompt the user for a movement command
    print("Enter movement command (F/B/L/R/S/Q): ")
    movement = keyboard.read_key()

    if movement.upper() == 'F':
        send_command('F')
        print("Moving forward...")
        while keyboard.is_pressed('F'):
            pass  # Wait for the 'F' key to be released
        send_command('S')  # Stop the car when the key is released
        print("Stopped.")
    elif movement.upper() == 'B':
        send_command('B')
        print("Moving backward...")
        while keyboard.is_pressed('B'):
            pass  # Wait for the 'B' key to be released
        send_command('S')  # Stop the car when the key is released
        print("Stopped.")
    elif movement.upper() == 'L':
        send_command('L')
        print("Turning left...")
        while keyboard.is_pressed('L'):
            pass  # Wait for the 'L' key to be released
        send_command('S')  # Stop the car when the key is released
        print("Stopped.")
    elif movement.upper() == 'R':
        send_command('R')
        print("Turning right...")
        while keyboard.is_pressed('R'):
            pass  # Wait for the 'R' key to be released
        send_command('S')  # Stop the car when the key is released
        print("Stopped.")
    elif movement.upper() == 'S':
        send_command('S')
        print("Stopping...")
    elif movement.upper() == 'Q':
        send_command('S')  # Stop the car before quitting
        print("Quitting...")
        break
    else:
        print("Invalid command. Please try again.")
ser.close()
