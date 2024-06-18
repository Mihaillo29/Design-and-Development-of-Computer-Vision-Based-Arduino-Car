import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
import random
import math
import geocoder
import os
import time
import serial

# Configure the serial port for communication with Arduino
ser = serial.Serial('COM5', 9600)

# Load the pre-trained model for object detection
model_obj = hub.load("./Mobilenetv2").signatures["default"]
colorcodes = {}

# Load the pre-trained segmentation model for lane detection
model_lane = load_model('LaneDetection/full_CNN_model.h5')

# Importing YOLOv4 model weights and config file
net1 = cv2.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net1.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model1 = cv2.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1 / 255, swapRB=True)

def draw_pothole(image, boxes, scores):
    pothole_detected = False  # Flag to indicate if a pothole is detected
    for (classid, score, box) in zip([0] * len(scores), scores, boxes):
        label = "pothole"
        x, y, w, h = box
        recarea = w * h
        area = image.shape[0] * image.shape[1]
        if (len(scores) != 0 and scores[0] >= 0.7):
            if ((recarea / area) <= 0.1 and box[1] < 600):
                pothole_detected = True  # Update flag if a pothole is detected
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(image, "%" + str(round(scores[0] * 100, 2)) + " " + label, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

    # Add "drive slowly" text if a pothole is detected
    if pothole_detected:
        cv2.putText(image, "Drive slowly", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image

g = geocoder.ip('me')
result_path = "pothole_coordinates"
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
i = 0
b = 0

# Define region of interest (ROI) vertices for lane detection
def get_roi_vertices(image):
    height, width = image.shape[:2]
    roi_bottom_left = (0, height)
    roi_bottom_right = (width, height)
    roi_top_left = (width // 2 - 50, height // 2 + 50)
    roi_top_right = (width // 2 + 50, height // 2 + 50)
    return np.array([[roi_bottom_left, roi_top_left, roi_top_right, roi_bottom_right]], dtype=np.int32)


# Function to calculate lane curvature (distance from the road)
def calculate_curvature(line_fit, y_eval):
    A, B = line_fit
    ym_per_pix = 30 / 720  # Example: 30 meters for 720 pixels height
    y_eval_meters = y_eval * ym_per_pix
    curvature = ((1 + (2 * A * y_eval_meters + B) ** 2) ** 1.5) / np.abs(2 * A)
    return curvature

# Function to process frame for lane detection
def process_frame(frame):
    global stop_sign_detected  # Declare stop_sign_detected as a global variable
    # Resize frame to match the input size of the lane detection model
    resized_frame = cv2.resize(frame, (160, 80))

    # Preprocess the resized frame (e.g., normalization, color conversion)
    preprocessed_frame = resized_frame  # Add any necessary preprocessing steps here

    # Perform lane detection using the pre-trained model
    if not stop_sign_detected:  # Check if stop sign is not detected
        lane_mask = model_lane.predict(np.expand_dims(preprocessed_frame, axis=0))
    else:
        # If stop sign is detected, set lane_mask to zeros to stop lane detection
        lane_mask = np.zeros_like(preprocessed_frame)

    # Post-process the lane mask to obtain the lane markings
    lane_mask = (lane_mask > 0.5).astype(np.uint8)
    lane_mask = cv2.resize(lane_mask[0], (frame.shape[1], frame.shape[0]))

    # Define region of interest
    mask = np.zeros_like(lane_mask)
    roi_vertices = get_roi_vertices(lane_mask)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_mask = cv2.bitwise_and(lane_mask, mask)

    # Find the contours of lane markings in the masked mask
    contours, _ = cv2.findContours(masked_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:  # Check if contours are found
        # Fit a polynomial to the lane markings
        lane_fit = np.polyfit(*contours[0][:, 0, :].T, 1)

        # Define the y-coordinate at which to evaluate the curvature (bottom of the frame)
        y_eval = frame.shape[0] - 1

        # Calculate the curvature of the lane
        curvature = calculate_curvature(lane_fit, y_eval)

        # Use Hough Line Transform to detect additional lane curves
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        # Process Hough lines to determine if they represent curves
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                if np.abs(theta - np.pi / 2) > np.pi / 6:  # Threshold angle for curve detection
                    if theta > np.pi / 2:
                        ser.write(b'G')  # Right curve detected
                    else:
                        ser.write(b'I')  # Left curve detected
                    break  # Only consider the first detected curve

        else:  # No Hough lines detected, send command based on lane curvature
            if curvature > 0.1:  # Curve to the right
                ser.write(b'G')
            elif curvature < -0.1:  # Curve to the left
                ser.write(b'I')
            else:  # Move forward by default
                ser.write(b'F')

    else:  # No contours found, move forward by default
        ser.write(b'S')

    # Draw lane markings on the frame
    line_image = np.zeros_like(frame)
    line_image[lane_mask == 1] = (0, 100, 0)
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    return result


# Function to draw bounding box for object detection
def drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color):
    im_height, im_width, _ = image.shape
    left, top, right, bottom = int(xmin * im_width), int(ymin * im_height), int(xmax * im_width), int(ymax * im_height)
    cv2.rectangle(image, (left, top), (right, bottom), color=color, thickness=1)
    FONT_SCALE = 5e-3
    THICKNESS_SCALE = 4e-3
    width = right - left
    height = bottom - top
    TEXT_Y_OFFSET_SCALE = 1e-2
    cv2.rectangle(
        image,
        (left, top - int(height * 6e-2)),
        (right, top),
        color=color,
        thickness=-1
    )
    cv2.putText(
        image,
        namewithscore,
        (left, top - int(height * TEXT_Y_OFFSET_SCALE)),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=min(width, height) * FONT_SCALE,
        thickness=math.ceil(min(width, height) * THICKNESS_SCALE),
        color=(255, 255, 255)
    )

# Function to draw bounding boxes on the image for object detection
def draw(image, boxes, classnames, scores):
    global stop_sign_detected  # Declare stop_sign_detected as a global variable
    boxesidx = tf.image.non_max_suppression(boxes, scores, max_output_size=20, score_threshold=0.2)
    car_detected = False  # Flag to indicate if a car is detected
    stop_sign_detected = False  # Flag to indicate if a stop sign is detected
    text_y_offset = 30  # Initial offset for text placement
    text_spacing = 30  # Spacing between lines of text
    for i in boxesidx:
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        classname = classnames[i].decode("ascii")
        if classname == 'Car' or classname == 'Vehicle' or classname == 'Land vehicle' or classname == 'Person' or classname == 'Stop sign' or classname == 'Man' or classname == 'Woman':
            if classname == 'Car' or classname == 'Vehicle' or classname == 'Land vehicle':
                car_detected = True  # Update flag if a car is detected
            elif classname == 'Stop sign':
                stop_sign_detected = True  # Update flag if a stop sign is detected
            if classname in colorcodes.keys():
                color = colorcodes[classname]
            else:
                c1 = random.randrange(0, 255, 30)
                c2 = random.randrange(0, 255, 25)
                c3 = random.randrange(0, 255, 50)
                colorcodes.update({classname: (c1, c2, c3)})
                color = colorcodes[classname]
            namewithscore = "{}:{}".format(classname, int(100 * scores[i]))
            # Draw bounding box and associated text
            drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color)
            # Update text y-position for the next line
            text_y_offset += text_spacing

    # Draw additional messages based on detected objects
    if car_detected:
        cv2.putText(image, "Drive carefully", (10, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        text_y_offset += text_spacing
    if stop_sign_detected:
        ser.write(b'S')  # Send stop command to Arduino
        cv2.putText(image, "Wait", (10, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image
# Function to read label names from obj.names file
class_name = []
with open(os.path.join("project_files", 'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Initialize video capture from webcam
video = cv2.VideoCapture("video1.mp4")
result_video = cv2.VideoWriter('result.mp4',
                               cv2.VideoWriter_fourcc(*'MJPG'),
                               10, (900, 700))

# Main program
def main():
    while True:
        _, img = video.read()
        if img is None:
            break
        img = cv2.resize(img, (900, 700))
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = tf.image.convert_image_dtype(img2, tf.float32)[tf.newaxis, ...]
        # Offload processing to GPU for object detection
        with tf.device('/GPU:0'):
            detection = model_obj(img2)
        # Convert detection results to numpy arrays for object detection
        result = {key: value.numpy() for key, value in detection.items()}
        
         # Draw bounding boxes on the image for object detection
        imagewithboxes = draw(img, result['detection_boxes'], result['detection_class_entities'],result["detection_scores"])
        
        # Analysis the stream with YOLOv4 model
        classes, scores, boxes = model1.detect(img, Conf_threshold, NMS_threshold)
        
        # Draw bounding boxes for pothole detection
        imagewithboxes = draw_pothole(imagewithboxes, boxes, scores)
        
        # Process frame for lane detection and send commands to Arduino
        processed_frame = process_frame(imagewithboxes)
        cv2.imshow('frame', processed_frame)
        result_video.write(processed_frame)
        # Check for key press for exiting both loops
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    # End of video capture and OpenCV windows
    video.release()
    result_video.release()
    ser.write(b'S')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
