from functions import *
import matplotlib as plt
import xlsxwriter
# Doc file video (khoảng 60 khung/s)
cap = cv2.VideoCapture("videos/video-nghieng.mp4")

# Khai bao tao Video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# outVideo = cv2.VideoWriter('output.avi', fourcc, 5, (1280, 720))

# new_width = 1096
# new_height = 800
new_width = 1112
new_height = 712
i = 0

blueLower = (90, 150, 0)
blueUpper = (120, 255, 255)

yellowLower = (20, 60, 60)
yellowUpper = (120, 255, 255)

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
    cv2.imwrite('images/video-nghieng/outImage%s.jpg' % (i), image)
    # cv2.imshow("Step1", image)
    # STEP 2: Find the coordinates of the laser pointer
    [x, y] = FindCenter(image)
    # STEP 3: Determine the coordinate of the Grid
    [verCoor, honCoor] = GridCoordinates(image)
    # STEP 4: Calculate the real coordinate of the laser pointer
    [x_real, y_real, img] = RealCoordinatesOfLaserPointer(image, x, y, verCoor, honCoor)
    # cv2.imshow("Detected Image", img)
    print(x_real, y_real)
    cv2.imwrite("Detected Images/video-nghieng/%s.jpg" %(i), img)
    # STEP 5: Calculate the distance change
    if i == 1:
        preX = x_real
        preY = y_real
    realCoor.append([x_real, y_real])


# After STEP 5, Save Data[X, Y] into Excel file
workbook = xlsxwriter.Workbook('Evaluation Results/video-nghieng.xlsx')
worksheet = workbook.add_worksheet()
col = 0
worksheet.write_row(0, 0, ['X', 'Y'])
for row, data in enumerate(realCoor):
    worksheet.write_row(row + 1, col, data)
workbook.close()

# Draw Graph

cv2.waitKey(0)
cv2.destroyAllWindows()