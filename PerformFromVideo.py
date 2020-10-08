from BW_functions import *
import matplotlib as plt
import xlsxwriter
import os


real_coor = []

final_width = 680
final_height = 249
direction = 'day08-10/2Y'

if not os.path.exists('Input_Images/' + direction):
    os.makedirs('Input_Images/' + direction)

if not os.path.exists('Detected_Images/' + direction):
    os.makedirs('Detected_Images/' + direction)

if not os.path.exists('Evaluation_Results_by_Excel/' + direction.split('/')[0]):
    os.makedirs('Evaluation_Results_by_Excel/' + direction.split('/')[0])


# Doc file video (khoảng 60 khung/s)
cap = cv2.VideoCapture('Input_Videos/' + direction + '.mp4')

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

    if (i == 1):
        new_width = np.shape(frame)[1]*3 if (np.shape(frame)[1] < 2560) else np.shape(frame)[1]
        new_height = np.shape(frame)[0]*3 if (np.shape(frame)[0] < 1922) else np.shape(frame)[0]


    original_image = cv2.resize(src=frame, dsize=(new_width, new_height))
    # cv2.imshow("Original image", original_image)
    cv2.imwrite(filename_read, original_image)


    # STEP 2: Detect WHITE frame
    white_frame = detect_white_frame(original_image)
    if white_frame.shape == original_image.shape:
        continue
    # cv2.imshow("Detected white frame", white_frame)
    # cv2.imwrite(filename_read, white_frame)

    # STEP 3: Find center point
    cX, cY = find_center_point(white_frame)
    # print(cX, cY)

    # STEP 4: Determine the coordinate of the Grid
    line1, line2, ver_coor = detect_grid_coodinate(white_frame)

    # STEP 5: Calculate the real coordinate of the laser pointer
    distance_x = calculate_real_coordinate_of_laser_pointer(cX, cY, ver_coor)
    print("Khoang cach: " + str(distance_x))

    # STEP 6: Draw and Save image
    font = cv2.FONT_HERSHEY_COMPLEX
    for j in range(10):
        cv2.line(white_frame, (line1[j][0], line1[j][1]), (line2[j][0], line2[j][1]), (0, 0, 0), 1)

    cv2.circle(white_frame, (cX, cY), 5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(white_frame, (cX, cY), (ver_coor[0], cY), (0, 0, 255), 1)
    cv2.putText(white_frame, str(distance_x) + 'cm', (int(cX / 2), cY - 10), font, 0.5, (255, 0, 0))

    final_images = cv2.resize(src=white_frame, dsize=(final_width, final_height))
    # cv2.imshow("Final image", final_images)
    cv2.imwrite(filename_write, final_images)

    # STEP 7: Calculate the distance change
    real_coor.append([i - 1, distance_x])

# print(np.shape(real_coor))

# After STEP 7, Save Data[X] into Excel file
workbook = xlsxwriter.Workbook('Evaluation_Results_by_Excel/' + direction + '.xlsx')
worksheet = workbook.add_worksheet()
col = 0
worksheet.write_row(0, 0, ['Time', 'X'])
for row, data in enumerate(real_coor):
    worksheet.write_row(row + 1, col, data)
workbook.close()


# Draw Graph
cv2.waitKey(0)
cv2.destroyAllWindows()