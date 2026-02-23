to run demo:
* only works in python 3.10, and have to use old version 0.10.5 of mediapipe

in command line:
1. Clone repository
2. py -3.10 -m venv venv
3. venv\Scripts\activate
4. python -m pip install --upgrade pip
5. pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
6. pip install mediapipe==0.10.5
7. pip install opencv-python pillow numpy
8. python webcam_gesture_demo.py
