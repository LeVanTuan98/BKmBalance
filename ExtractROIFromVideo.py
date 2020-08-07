from BW_functions import *
import matplotlib as plt
import xlsxwriter
# Doc file video (khoảng 60 khung/s)
cap = cv2.VideoCapture("Input_Videos/video7-8_1.mp4")

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
    fileNameImage = 'Input_Images/video7-8/video7-8_1/frame' + str(i) + '.jpg'
    print(fileNameImage)
    cv2.imwrite(fileNameImage, image)



cv2.waitKey(0)
cv2.destroyAllWindows()