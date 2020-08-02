from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import matplotlib.pyplot as plt
from functions import *

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    minX = min(tl[0], bl[0])
    maxX = max(tr[0], br[0])

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    minY = min(tl[1], tr[1])
    maxY = max(bl[1], br[1])
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped, minX, maxX, minY, maxY


whiteLower = (0, 0, 245)
whiteUpper = (255, 150, 255)

blueLower = (100, 100, 100)
blueUpper = (120, 255, 255)

screenCntWhite = 0



# read image
image = cv2.imread("images/image_White-Blue/3.jpg")

# resize image
orig = image
# ratio = image.shape[0] / 500.0  # Chiều cao ảnh chuẩn hóa (chia cho 500)
# image = imutils.resize(image, height = 500)

# convert the image to grayscale, blur it, and find edges
# in the image
blurred = cv2.GaussianBlur(image, (5, 5), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

maskBlue = cv2.inRange(hsv, blueLower, blueUpper)

maskBlue = cv2.erode(maskBlue, None, iterations=2)
maskBlue = cv2.dilate(maskBlue, None, iterations=2)

# show the original image and the edge detected image
print("STEP 1: Color Detection BLUE")
# cv2.imshow("Image", image)
# cv2.imshow("Color", maskBlue)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Step 2: Finding Contours
# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cntsBlue = cv2.findContours(maskBlue.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cntsBlue = imutils.grab_contours(cntsBlue)
cntsBlue = sorted(cntsBlue, key=cv2.contourArea, reverse=True)[:4]

# loop over the contours
for roiBlue in cntsBlue:
    peri = cv2.arcLength(roiBlue, True)
    approx = cv2.approxPolyDP(roiBlue, 0.02 * peri, True)

    maskBlue = np.zeros_like(orig)
    cv2.drawContours(maskBlue, [approx], 0, (255, 255, 255), -1)  # Draw filled contour in mask

    warpedBlue = np.zeros_like(orig)  # Extract out the object and place into output image
    warpedBlue[maskBlue == 255] = orig[maskBlue == 255]
    origBlue = warpedBlue

    blurredWar = cv2.GaussianBlur(warpedBlue, (5, 5), 0)
    hsvWar = cv2.cvtColor(warpedBlue, cv2.COLOR_BGR2HSV)

    maskWar = cv2.inRange(hsvWar, whiteLower, whiteUpper)

    maskWar = cv2.erode(maskWar, None, iterations=2)
    maskWar = cv2.dilate(maskWar, None, iterations=2)  # Đang là ảnh Gray với 2 mức xám 0 và 255

    # calculate moments of binary image
    M = cv2.moments(maskWar)

    # calculate x,y coordinate of center
    cXTemp = int(M["m10"] / M["m00"])
    cYTemp = int(M["m01"] / M["m00"])

    warpedBlue = cv2.medianBlur(origBlue, 5)
    grayImage = cv2.cvtColor(warpedBlue, cv2.COLOR_BGR2GRAY)
    valueThreshold, binaryImage = cv2.threshold(grayImage, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cntsWhite = cv2.findContours(binaryImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cntsWhite = imutils.grab_contours(cntsWhite)
    cntsWhite = sorted(cntsWhite, key=cv2.contourArea, reverse=True)[:4]

    # loop over the contours
    for roiWhite in cntsWhite:
        # approximate the contour
        peri = cv2.arcLength(roiWhite, True)
        approx = cv2.approxPolyDP(roiWhite, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            _, minX, maxX, minY, maxY = four_point_transform(origBlue, approx.reshape(4, 2))
            if minX < cXTemp < maxX and minY < cYTemp < maxY:
                screenCntWhite = approx
                x = int(cXTemp - minX)
                y = int(cYTemp - minY)
                break
            else:
                continue

    if screenCntWhite.all() != 0:
        break

if screenCntWhite.all() == 0:
    print("Không tìm thấy vùng màu trắng")
else:
    warpedWhite, _, _, _, _ = four_point_transform(orig, screenCntWhite.reshape(4, 2))

print("STEP 2: Rounding white paper")
new_width = 1112
new_height = 712

image = warpedWhite
cv2.imshow("Rounding white paper", image)
## tim tâm
# Improve the algorithm finding the centre laser
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # color -> gray
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.threshold(blurred, 240 ,255, cv2.THRESH_BINARY_INV)[1]
canny = cv2.Canny(thresh, 50, 255, 1)
delta = 30
mask_laser = canny[y - delta: y + delta, x - delta: x + delta]

# Focusing on [top right bottom left] of red region
[top, bottom, left, right] = FindTRBL(mask_laser)
cX = x + int((right + left)/2) - delta
cY = y + int((top + bottom)/2) - delta









print('STEP 3: Determine the coordinate of the Grid')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# threshold
th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
cv2.imshow("thresh", threshed)
# findcontours
cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
# filter by area
number_dot_per_line = 10
S_min = 3
S_max = 300
xcnts = []
coor_x = []
coor_y = []
for cnt in cnts:
    if S_min < cv2.contourArea(cnt) < S_max:
        xcnts.append(cnt)
        # calculate moments of binary image
        M = cv2.moments(cnt)

        # calculate x,y coordinate of center
        grid_x = int(M["m10"] / M["m00"])
        grid_y = int(M["m01"] / M["m00"])
        coor_x.append(grid_x)
        coor_y.append(grid_y)
        cv2.circle(image, (grid_x, grid_y), 1, (255, 0, 0), 1, cv2.LINE_AA)

number_dots = len(xcnts)
if number_dots > number_dot_per_line * 2:
    print('So diem lon hon 20')
print(x, y)
print(coor_x)
print(coor_y)
cv2.circle(image, (cX, cY), 5, (0, 0, 255), 1, cv2.LINE_AA)

line1 = []
line2 = []
ver_coor = []
for i in range(number_dot_per_line):
    x1 = coor_x[0]
    y1 = coor_y[0]
    coor_x.pop(0)
    coor_y.pop(0)
    coor_temp = abs(coor_x - np.ones(np.size(coor_x))*x1)
    min_temp = min(coor_temp)
    for j in range(len(coor_temp)):
        if coor_temp[j] == min_temp:
            index_x2 = j
            break
    x2 = coor_x[index_x2]
    y2 = coor_y[index_x2]
    coor_x.pop(index_x2)
    coor_y.pop(index_x2)
    xtb = int((x1 + x2)/2)
    ver_coor.append(xtb)
    if y1 < y2:
        line1.append([xtb, y1])
        line2.append([xtb, y2])
    else:
        line2.append([xtb, y1])
        line1.append([xtb, y2])
print(line1)
print(line2)

for i in range(number_dot_per_line):
    cv2.line(image, (line1[i][0], line1[i][1]), (line2[i][0], line2[i][1]), (0, 0, 255), 1)
cv2.imshow("mask_grid", image)

# STEP 4: Calculate the real coordinate of the laser pointer and save image
ver_coor.sort()
[x_real, img] = RealCoordinatesOfLaserPointer(image, cX, cY, ver_coor)
print(x_real)
cv2.imshow("Calculated Image", img)
cv2.imwrite('Detected Images/image_White-Blue/3.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()