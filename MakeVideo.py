import cv2
import numpy as np
import glob

img_array = []


# for filename in glob.glob('Detected Images/video13-6/*.jpg'):
for i in range(1, 333):
    filename = 'Detected Images/video-nghieng/%s.jpg' %(i)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
    print(filename)



out = cv2.VideoWriter('Reconstruction Video/video-nghieng.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()