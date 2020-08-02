from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import matplotlib.pyplot as plt


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

def detect_white_frame_laser_poiter(original_image):
    image = original_image.copy()

    whiteLower = (0, 0, 245)
    whiteUpper = (255, 150, 255)

    blueLower = (100, 100, 100)
    blueUpper = (120, 255, 255)

    screenCntWhite = 0

    # convert the image to grayscale, blur it, and find edges
    # in the image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    maskBlue = cv2.inRange(hsv, blueLower, blueUpper)

    maskBlue = cv2.erode(maskBlue, None, iterations=2)
    maskBlue = cv2.dilate(maskBlue, None, iterations=2)

    # show the original image and the edge detected image
## STEP 1: Color Detection - BLUE
    # cv2.imshow("Image", image)
    # cv2.imshow("Color", maskBlue)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# 1.1 Finding Contours
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cntsBlue = cv2.findContours(maskBlue.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cntsBlue = imutils.grab_contours(cntsBlue)
    cntsBlue = sorted(cntsBlue, key=cv2.contourArea, reverse=True)[:4]

    # loop over the contours
    for roiBlue in cntsBlue:
        peri = cv2.arcLength(roiBlue, True)
        approx = cv2.approxPolyDP(roiBlue, 0.02 * peri, True)

        maskBlue = np.zeros_like(original_image)
        cv2.drawContours(maskBlue, [approx], 0, (255, 255, 255), -1)  # Draw filled contour in mask

        warpedBlue = np.zeros_like(original_image)  # Extract out the object and place into output image
        warpedBlue[maskBlue == 255] = original_image[maskBlue == 255]
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
        print("Khong tim thay vung mau trang")
    else:
        warpedWhite, _, _, _, _ = four_point_transform(original_image, screenCntWhite.reshape(4, 2))

## STEP 2: Find centre of laser poiter (x, y)
    # Improve the algorithm finding the centre laser
    frame = warpedWhite
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # color -> gray
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)[1]
    canny = cv2.Canny(thresh, 50, 255, 1)
    delta = 30
    mask_laser = canny[y - delta: y + delta, x - delta: x + delta]

    # Focusing on [top right bottom left] of red region
    [top, bottom, left, right] = FindTRBL(mask_laser)
    cX = x + int((right + left) / 2) - delta
    cY = y + int((top + bottom) / 2) - delta
    return warpedWhite, cX, cY

### Ham tim Top - Right - Bottom - Left
def FindTRBL(values):
    # Input: Ma tran can tim T - R - B - L
    # Output: Gia tri cua Tmost - Rmost - Bmost - Lmost
    leftmost = 0
    rightmost = 0
    topmost = 0
    bottommost = 0
    temp = 0
    for i in range(np.size(values, 1)):
        col = values[:, i]
        if np.sum(col) != 0.0:
            rightmost = i
            if temp == 0:
                leftmost = i
                temp = 1
    for j in range(np.size(values, 0)):
        row = values[j, :]
        if np.sum(row) != 0.0:
            bottommost = j
            if temp == 1:
                topmost = j
                temp = 2
    return [topmost, bottommost, leftmost, rightmost]

def detect_grid_coodinate(warped_image):
    image = warped_image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # threshold
    th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
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
# Tim tam cua cac nut luoi dua vao dieu kien dien tich [S_min, S_max]
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
            cv2.circle(image, (grid_x, grid_y), 2, (255, 0, 0), 2, cv2.LINE_AA)

    if len(xcnts) != number_dot_per_line * 2:
        print('Khong tim thay dung so diem nut!!!')
    # cv2.circle(image, (cX, cY), 5, (0, 0, 255), 1, cv2.LINE_AA)

    line1 = [] # toa do cua cac diem hang tren
    line2 = [] # toa do cac diem hang duoi
    ver_coor = [] # toa do cac diem theo truc x
# Sap xep cac nut luoi theo tung cap voi cung toa do x
    for i in range(number_dot_per_line):
        x1 = coor_x[0]
        y1 = coor_y[0]
        coor_x.pop(0)
        coor_y.pop(0)
        coor_temp = abs(coor_x - np.ones(np.size(coor_x)) * x1)
        min_temp = min(coor_temp)
        for j in range(len(coor_temp)):
            if coor_temp[j] == min_temp:
                index_x2 = j
                break
        x2 = coor_x[index_x2]
        y2 = coor_y[index_x2]
        coor_x.pop(index_x2)
        coor_y.pop(index_x2)
        xtb = int((x1 + x2) / 2)
        ver_coor.append(xtb)
        if y1 < y2:
            line1.append([xtb, y1])
            line2.append([xtb, y2])
        else:
            line2.append([xtb, y1])
            line1.append([xtb, y2])
    print(line1)
    print(line2)
    print(ver_coor)
    ver_coor.sort()
    return line1, line2, ver_coor
def calculate_real_coordinate_of_laser_pointer(cX, cY, ver_coor):
    # Input: x, y: Toa do tam cua diem laser
    #        verCoor: Toa do cua cac truc doc
    # Output: [x_real]: Toa do thuc te cua diem laser theo x

    # Kich thuoc thuc cua khung giay [Dai x Rong]
    # real_size = [13.9, 8.9]
    real_size = [1, 2.5]

    delta_x = np.diff(ver_coor)
    # delta_y = np.diff(honCoor)


# Tinh khoang cach tu tam den duong dau tien ben trai
# Neu tam nam giua 2 cot => tinh ty le khoang cach tu tam den cot ben trai gan nhat + so cot o giua
    delta = ver_coor - np.ones(np.size(ver_coor))*cX
    minValue = min(abs(delta))
    for i in range(len(delta) - 1):
        if np.sign(delta[i]) != np.sign(delta[i + 1]):
            # Tinh khoang cach den i - cot ben trai gan nhat
            scale_x = real_size[0]/(delta_x[i])
            x_real = round((cX - ver_coor[i])*scale_x + i, 2)
        if abs(delta[i]) == minValue:
            if minValue == 0:
                x_real = i
            break
    return x_real