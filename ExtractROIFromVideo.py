from functions import *
import matplotlib as plt
import xlsxwriter
# Doc file video (khoảng 60 khung/s)
cap = cv2.VideoCapture("videos/video16-6/May Tuan/video1_Trim.mp4")

# Khai bao tao Video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# outVideo = cv2.VideoWriter('output.avi', fourcc, 5, (1280, 720))

new_width = 1112
new_height = 712
i = 0


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
    fileNameImage = 'images/video16-6/may tuan/video1/outImage%s.jpg' % (i)
    print(fileNameImage)
    cv2.imwrite(fileNameImage, image)



cv2.waitKey(0)
cv2.destroyAllWindows()