import cv2
import numpy as np
import glob

img_array = []


for filename in glob.glob('Detected_Images/video7-8/video7-8_1/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
    print(filename)

out = cv2.VideoWriter('Reconstruction_Videos/video7-8_1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()