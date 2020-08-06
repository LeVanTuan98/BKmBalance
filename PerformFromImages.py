import matplotlib as plt
import xlsxwriter
from BW_functions import *
real_coor = []

for i in range(1, 321):
    filename_read = 'images/video6-8/outImage%s.jpg' % (i)
    filename_write = 'Detected Images/video6-8/detectedImage%s.jpg' %(i)
    # filename_read = 'images/video4-8/frame1.jpg'
    # filename_write = 'Detected Images/video4-8/detected1.jpg'

    print(filename_read)
    print(filename_write)

# STEP 1: Load image
    original_image = cv2.imread(filename_read)
    # cv2.imshow("Original image", original_image)

# STEP 2: Detect WHITE frame and the coordinate of laser pointer
    white_frame = detect_white_frame(original_image)
    # cv2.imshow("Detected white frame", white_frame)

# STEP 3: Find center point
    cX, cY = find_center_point(white_frame)
    print(cX, cY)

# STEP 4: Determine the coordinate of the Grid
    line1, line2, ver_coor = detect_grid_coodinate(white_frame)

# STEP 5: Calculate the real coordinate of the laser pointer
    distance_x = calculate_real_coordinate_of_laser_pointer(cX, cY, ver_coor)

# STEP 6: Draw and Save image
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.circle(white_frame, (cX, cY), 5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(white_frame, (cX, cY), (ver_coor[0], cY), (0, 0, 255), 1)
    cv2.putText(white_frame, str(distance_x) + 'cm', (ver_coor[0], cY - 30), font, 1, (255, 0, 0))
    for i in range(10):
        cv2.line(white_frame, (line1[i][0], line1[i][1]), (line2[i][0], line2[i][1]), (255, 0, 0), 1)
    cv2.imshow("Final image", white_frame)
    cv2.imwrite(filename_write, white_frame)

# STEP 6: Calculate the distance change
    # if i == 1
    #     preX = x_real
    #     preY = y_real
    real_coor.append([distance_x])

print(np.shape(real_coor))

# # After STEP 5, Save Data[X, Y] into Excel file
# workbook = xlsxwriter.Workbook('Evaluation Results/video13-6.xlsx')
# worksheet = workbook.add_worksheet()
# col = 0
# worksheet.write_row(0, 0, ['X', 'Y'])
# for row, data in enumerate(realCoor):
#     worksheet.write_row(row + 1, col, data)
# workbook.close()

# Draw Graph

cv2.waitKey(0)
cv2.destroyAllWindows()