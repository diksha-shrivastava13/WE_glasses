import numpy as np
import dlib
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import statistics

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img = dlib.load_rgb_image(path)
plt.imshow(img)

rect = detector(img)[0]
sp = predictor(img, rect)
landmarks = np.array([[p.x, p.y] for p in sp.parts()])

nose_bridge_x = []
nose_bridge_y = []

for i in [28, 29, 30, 31, 34, 35]:
    nose_bridge_x.append(landmarks[i][0])
    nose_bridge_y.append(landmarks[i][1])

x_max = max(nose_bridge_x)
y_max = max(nose_bridge_y)

x_min = min(nose_bridge_x)
y_min = min(nose_bridge_y)

img2 = Image.open(path)
img2 = img2.crop((x_min, y_min, x_max, y_max))
plt.imshow(img2)

img_blur = cv2.GaussianBlur(np.array(img2), (3, 3), sigmaX=0, sigmaY=0)
