from BW_functions import *
import matplotlib as plt
import xlsxwriter
# Doc file video (khoảng 60 khung/s)
cap = cv2.VideoCapture("videos/video6-8_2.mp4")

# Khai bao tao Video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# outVideo = cv2.VideoWriter('output.avi', fourcc, 5, (1280, 720))

new_width = 848*3
new_height = 480*3
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
    image = cv2.resize(src=frame, dsize=(new_width, new_height))
    fileNameImage = 'images/video6-8/outImage%s.jpg' % (i)
    print(fileNameImage)
    cv2.imwrite(fileNameImage, image)



cv2.waitKey(0)
cv2.destroyAllWindows()