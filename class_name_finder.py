############################################ I Used this to find the classes of Mobilenetv2 ########################################################
import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub

model = hub.load("./Mobilenetv2").signatures["default"]

# Capture video
video = cv2.VideoCapture("p.mp4")

class_names = set()

while True:
    _, img = video.read()
    img = cv2.resize(img, (900, 700))
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)[tf.newaxis, ...]
    detection = model(img2)
    result = {key: value.numpy() for key, value in detection.items()}
    class_entities = result['detection_class_entities']
    for class_entity in class_entities:
        class_name = class_entity.decode("ascii")
        class_names.add(class_name)
    
    cv2.imshow("image", img)
    key = cv2.waitKey(27) & 0xFF
    if key == ord('q'):  # If 'q' is pressed
        break  # Quit the loop and exit the program

cv2.destroyAllWindows()  # Close all OpenCV windows

print("Class names available in the model:")
for class_name in class_names:
    print(class_name)
