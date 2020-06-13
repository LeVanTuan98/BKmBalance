from functions import *
import matplotlib as plt
import xlsxwriter
# Doc file video (khoảng 60 khung/s)
cap = cv2.VideoCapture("videos/video-nghieng.mp4")

# Khai bao tao Video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# outVideo = cv2.VideoWriter('output.avi', fourcc, 5, (1280, 720))

new_width = 1096
new_height = 800
i = 0

realCoor = []

while True:
    # Read a new frame
    ok, frame = cap.read()
    if not ok:
        # Neu khong doc duoc tiep thi out
        break
    else:
        i += 1

    # STEP 1: Detect ROI
    image = DetectFrame(frame, i)
    image = cv2.resize(src=image, dsize=(new_width, new_height))
    fileNameImage = 'images/video-nghieng/outImage%s.jpg' % (i)
    cv2.imwrite(fileNameImage, image)



cv2.waitKey(0)
cv2.destroyAllWindows()