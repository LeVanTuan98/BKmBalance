from functions import *
import matplotlib as plt
import xlsxwriter


realCoor = []
for i in range(1, 333):
    filename = 'images/video13-6/outImage%s.jpg' %(i)
    print(filename)
    # STEP 1: Load image
    image = cv2.imread(filename)
    # cv2.imshow("Step1", image)
    # STEP 2: Find the coordinates of the laser pointer
    [x, y] = FindCenter(image)
    print(x, y)
    # STEP 3: Determine the coordinate of the Grid
    [verCoor, honCoor] = GridCoordinates(image)
    print(verCoor)
    print(honCoor)
   # STEP 4: Calculate the real coordinate of the laser pointer and save image
    [x_real, y_real, img] = RealCoordinatesOfLaserPointer(image, x, y, verCoor, honCoor)
    print(x_real, y_real)
    # cv2.imshow("Calculated Image", img)
    cv2.imwrite("Detected Images/video13-6/%s.jpg" %(i), img)
    #STEP 5: Calculate the distance change
    # if i == 1:
    #     preX = x_real
    #     preY = y_real
    realCoor.append([x_real, y_real])

print(np.shape(realCoor))

# After STEP 5, Save Data[X, Y] into Excel file
workbook = xlsxwriter.Workbook('Evaluation Results/video13-6.xlsx')
worksheet = workbook.add_worksheet()
col = 0
worksheet.write_row(0, 0, ['X', 'Y'])
for row, data in enumerate(realCoor):
    worksheet.write_row(row + 1, col, data)
workbook.close()

# Draw Graph

cv2.waitKey(0)
cv2.destroyAllWindows()