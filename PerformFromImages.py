from functions import *
import matplotlib as plt
import xlsxwriter
new_width = 860
new_height = 250

realCoor = []
for i in range(1, 333):
    i = 113
    # filename = 'images/video-nghieng/outImage%s.jpg' %(i)
    filename = 'images/newGridTrim.jpg'
    print(filename)
    # STEP 1: Load image
    image = cv2.imread(filename)
    image = cv2.resize(src=image, dsize=(new_width, new_height))
    cv2.imshow("Step1", image)
    # STEP 2: Find the coordinates of the laser pointer
    [x, y] = FindCenter(image)
    print(x, y)
    # STEP 3: Determine the coordinate of the Grid
    verCoor = GridCoordinates(image)
    print(verCoor)
   # STEP 4: Calculate the real coordinate of the laser pointer and save image
    [x_real, img] = RealCoordinatesOfLaserPointer(image, x, y, verCoor)
    print(x_real)
    cv2.imshow("Calculated Image", img)
    break
    cv2.imwrite("Detected Images/video13-6/%s.jpg" %(i), img)
    #STEP 5: Calculate the distance change
    # if i == 1
    #     preX = x_real
    #     preY = y_real
    realCoor.append([x_real])

print(np.shape(realCoor))

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