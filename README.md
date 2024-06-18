### Design and Development of Computer Vision Based Arduino Car

This repository contains the code and documentation for our research on an Arduino-based autonomous vehicle. Our project integrates lane detection and object recognition to create a driving assistance system. Key features include:

- **Lane Detection**: Utilizes a fully convolutional neural network (CNN) to output an image of a predicted lane, improving upon traditional lane detection methods.
- **Object Recognition**: Employs YOLO and SSD algorithms to detect vehicles, pedestrians, road signs, and potholes.
- **Hardware Integration**: Combines an Arduino Uno, HC05 Bluetooth module, motor control components, and a camera for real-time image processing and decision-making.

Please see my Project Report [here](https://www.alborearpress.com/jecra/p/JECRA1.pdf).


Our model has been rigorously tested in controlled environments, demonstrating significant advancements in autonomous driving technology. This project aims to enhance the safety and efficiency of self-driving cars by maintaining lane discipline and avoiding obstacles.

You can download the full training set of images used [here](https://www.dropbox.com/s/rrh8lrdclzlnxzv/full_CNN_train.p?dl=0) and the full set of labels [here](https://www.dropbox.com/s/ak850zqqfy6ily0/full_CNN_labels.p?dl=0).

![Github upload](https://github.com/Mihaillo29/Design-and-Development-of-Computer-Vision-Based-Arduino-Car/assets/117961472/b1a69e17-c669-4755-b2b9-0f3c71b93578)

## How the Code Works 
> Run ```main.py``` after setting up your arduino car

## Important Notes
It is worth noting that this project was created as a demonstration and its effectiveness depends on the video provided. The part image is very simple and line recognition may not be possible in poor or unstable lighting conditions.
If you use our work please contact us

## Acknowledgements
[Gauri Gandhi](https://github.com/Candyxoxo)
