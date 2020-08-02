from functions import *
import matplotlib as plt
import xlsxwriter
# Doc file video (khoảng 60 khung/s)
cap = cv2.VideoCapture("videos/video27-6/MayTuan1.mp4")

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
    # image = DetectFrame(frame, i)
    image = cv2.resize(src=frame, dsize=(1928, 1080))

    cv2.imshow("Frame", image)
    j = '{0:04}'.format(i)
    fileNameImage = 'images/video27-6/MayTuan/frame%s.jpg' %(j)
    print(fileNameImage)
    cv2.imwrite("images/test.jpg", image)
    break

cv2.waitKey(0)
cv2.destroyAllWindows()