import tkinter as tk
from tkinter import messagebox
import subprocess
from PIL import Image, ImageTk

def run_auto_pilot():
    try:
        subprocess.Popen(["python", "auto.py"])
    except FileNotFoundError:
        messagebox.showerror("Error", "auto.py script not found.")

def run_manual():
    try:
        subprocess.Popen(["python", "manual.py"])
    except FileNotFoundError:
        messagebox.showerror("Error", "manual.py script not found.")

def on_enter(e):
    e.widget.config(bg="#AEB6BF", fg="#FFFFFF")

def on_leave(e):
    e.widget.config(bg="#D5DBDB", fg="#333333")

def main():
    root = tk.Tk()
    root.title("Car Control System")
    root.geometry("500x400")  # Adjust size as needed
    root.configure(bg="#D5DBDB")  # Set background color

    # Load images for buttons
    auto_image = Image.open("auto.png")
    auto_image = auto_image.resize((120, 120), Image.BILINEAR)
    auto_photo = ImageTk.PhotoImage(auto_image)

    manual_image = Image.open("manual.png")
    manual_image = manual_image.resize((120, 120), Image.BILINEAR)
    manual_photo = ImageTk.PhotoImage(manual_image)

    # Create Auto button
    auto_button = tk.Button(root, text="Auto", font=("Helvetica", 14, "bold"), image=auto_photo, compound=tk.TOP, command=run_auto_pilot, borderwidth=0, bg="#D5DBDB", activebackground="#AEB6BF")
    auto_button.image = auto_photo
    auto_button.place(relx=0.25, rely=0.5, anchor=tk.CENTER)
    auto_button.bind("<Enter>", on_enter)
    auto_button.bind("<Leave>", on_leave)

    # Create Manual button
    manual_button = tk.Button(root, text="Manual", font=("Helvetica", 14, "bold"), image=manual_photo, compound=tk.TOP, command=run_manual, borderwidth=0, bg="#D5DBDB", activebackground="#AEB6BF")
    manual_button.image = manual_photo
    manual_button.place(relx=0.75, rely=0.5, anchor=tk.CENTER)
    manual_button.bind("<Enter>", on_enter)
    manual_button.bind("<Leave>", on_leave)

    # Title label
    title_label = tk.Label(root, text="Car Control System", font=("Helvetica", 24, "bold"), bg="#D5DBDB", fg="#333333")
    title_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

    root.mainloop()

if __name__ == "__main__":
    main()
