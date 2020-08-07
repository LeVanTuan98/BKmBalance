from BW_functions import *
import matplotlib as plt
import xlsxwriter


# Khai bao tao Video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# outVideo = cv2.VideoWriter('output.avi', fourcc, 5, (1280, 720))


real_coor = []
new_width = 848*3
new_height = 480*3

final_width = 680
final_height = 249

direction = 'video7-8/video7-8_4'
# Doc file video (khoảng 60 khung/s)
cap = cv2.VideoCapture("Input_Videos/video7-8_4.mp4")

i = 0

while True:
    # Read a new frame
    ok, frame = cap.read()
    if not ok:
        # Neu khong doc duoc tiep thi out
        break
    else:
        i += 1

    filename_read = 'Input_Images/' + direction + '/frame' + str('{0:04}'.format(i)) + '.jpg'
    filename_write = 'Detected_Images/' + direction + '/detected' + str('{0:04}'.format(i)) + '.jpg'

    # STEP 1: Load image
    print(filename_read)
    # original_image = frame
    original_image = cv2.resize(src=frame, dsize=(new_width, new_height))
    # cv2.imshow("Original image", original_image)
    cv2.imwrite(filename_read, original_image)

    # STEP 2: Detect WHITE frame
    white_frame = detect_white_frame(original_image)
    # cv2.imshow("Detected white frame", white_frame)
    # cv2.imwrite(filename_read, white_frame)

    # STEP 3: Find center point
    cX, cY = find_center_point(white_frame)
    # print(cX, cY)

    # STEP 4: Determine the coordinate of the Grid
    line1, line2, ver_coor = detect_grid_coodinate(white_frame)

    # STEP 5: Calculate the real coordinate of the laser pointer
    distance_x = calculate_real_coordinate_of_laser_pointer(cX, cY, ver_coor)
    print(distance_x)

    # STEP 6: Draw and Save image
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.circle(white_frame, (cX, cY), 5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(white_frame, (cX, cY), (ver_coor[0], cY), (0, 0, 255), 1)
    cv2.putText(white_frame, str(distance_x) + 'cm', (ver_coor[0], cY - 30), font, 1, (255, 0, 0))
    for j in range(10):
        cv2.line(white_frame, (line1[j][0], line1[j][1]), (line2[j][0], line2[j][1]), (255, 0, 0), 1)
    # cv2.imshow("Final image", white_frame)
    final_images = cv2.resize(src=white_frame, dsize=(final_width, final_height))
    cv2.imwrite(filename_write, final_images)

    # STEP 6: Calculate the distance change
    # if i == 1
    #     preX = x_real
    #     preY = y_real
    real_coor.append([distance_x])


# After STEP 5, Save Data[X, Y] into Excel file
workbook = xlsxwriter.Workbook('Evaluation_Results_by_Excel/' + direction + '.xlsx')
worksheet = workbook.add_worksheet()
col = 0
worksheet.write_row(0, 0, ['X'])
for row, data in enumerate(real_coor):
    worksheet.write_row(row + 1, col, data)
workbook.close()


# Draw Graph

cv2.waitKey(0)
cv2.destroyAllWindows()