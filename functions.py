import numpy as np
import cv2
import imutils
# Kich thuoc thuc cua khung giay [Dai x Rong]
# real_size = [13.9, 8.9]
real_size = [1, 2.5]
# So duong ke thuc te nam
# doc (numVer) va nam ngang (honVer)
numVer = 11
honVer = 2

# Khoang mau de loc Frame : DetectFrame
blackLower = (0, 0, 0)
blackUpper = (180, 120, 120)

# Khoang mau de loc laser pointer: FindCenter
whiteLower = (0, 100, 245)
whiteUpper = (255, 255, 255)
### Hàm tách Frame
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
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
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
    return warped
def DetectFrame(frame, i):
    # blackLower = (0, 0, 0)
    # blackUpper = (180, 120, 120)


    ratio = frame.shape[0] / 500.0  # Chiều cao ảnh chuẩn hóa(chia cho 500)
    orig = frame.copy()
    frame = imutils.resize(frame, height=500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # edged = cv2.Canny(gray, 75, 200)

    mask = cv2.inRange(hsv, blackLower, blackUpper)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # show the original image and the edge detected image
    # print("STEP 1: Edge Detection")
    # cv2.imshow("Image", frame)
    # cv2.imshow("Edged", edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Step 2: Finding Contours
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    if len(approx) != 4:
        print("No Find the Paper: ", i)

    # show the contour (outline) of the piece of paper
    # print("STEP 2: Find contours of paper")
    # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    # cv2.imshow("Outline", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Step 3: Apply a Perspective Transform & Threshold
    # apply the four point transform to obtain a top-down
    # view of the original image
    try:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        # convert the warped image to grayscale, then threshold it
        # to give it that 'black and white' paper effect
        # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # T = threshold_local(warped, 11, offset=10, method="gaussian")
        # warped = (warped > T).astype("uint8") * 255
        # show the original and scanned images
        # fileNameImage = 'outImage%s.jpg' % (i)
        # print(fileNameImage)
        # cv2.imwrite(fileNameImage, warped)
    except:
        print("Error in Frame: ", i)

    return warped
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
### Ham xac dinh toa do trung tam cua laser Pointer
# def FindCenter(image):
#     # Input: Anh dau vao da tach khung
#     # Output: Toa do (x, y) cua laser pointer
#     img = image.copy()
#
#     # whiteLower = (0, 0, 245)
#     # whiteUpper = (255, 255, 255)
#
#     blurredWar = cv2.GaussianBlur(img, (5, 5), 0)
#     hsvWar = cv2.cvtColor(blurredWar, cv2.COLOR_BGR2HSV)
#
#     maskWar = cv2.inRange(hsvWar, whiteLower, whiteUpper)
#     cv2.imshow("hsvWar1", maskWar)
#     maskWar = cv2.dilate(maskWar, None, iterations=2)  # Đang là ảnh Gray với 2 mức xám 0 và 255
#     maskWar = cv2.erode(maskWar, None, iterations=2)
#
#     cv2.imshow("hsvWar2", maskWar)
#     # convert the grayscale image to binary image
#     ret, thresh = cv2.threshold(maskWar, 127, 255, 0)
#     cv2.imshow("hsvWar3", thresh)
#     # calculate moments of binary image
#     # M = cv2.moments(thresh)
#     #
#     # # calculate x,y coordinate of center
#     # cX = int(M["m10"] / M["m00"])
#     # cY = int(M["m01"] / M["m00"])
#     # Focusing on [top right bottom left] of red region
#     [top, bottom, left, right] = FindTRBL(thresh)
#     cX = int((right + left)/2)
#     cY = int((top + bottom)/2)
#
#     return cX, cY


def RealCoordinatesOfLaserPointer(image, x, y, verCoor):
    # Input: x, y: Toa do cua diem laser
    #        verCoor: Toa do cua cac truc doc
    #        honCoor: Toa do cua ca truc ngang
    #        scale: Ty le quy doi (cm/pixels)
    # Output: [x_real, y_real]: Toa do thuc te cua diem laser

    img = image.copy()
    # size_y = np.size(img, 0)
    # size_x = np.size(img, 1)

    delta_x = np.diff(verCoor)
    # delta_y = np.diff(honCoor)

    font = cv2.FONT_HERSHEY_COMPLEX
    # Duyet theo hang ngang
    delta = verCoor - np.ones(np.size(verCoor))*x
    minValue = min(abs(delta))
    for i in range(len(delta) - 1):
        if np.sign(delta[i]) != np.sign(delta[i + 1]):
            # Tinh khoang cach den i
            scale_x = real_size[0]/(delta_x[i])
            x_real = round((x - verCoor[i])*scale_x + i, 2)
            cv2.line(img, (x, y), (verCoor[0], y), (0, 0, 255), 1)
            cv2.putText(img, str(x_real) + 'cm', (verCoor[0], y - 30), font, 1, (0, 255, 255))
        if abs(delta[i]) == minValue:
            if minValue == 0:
                x_real = i
            break
    ### Duyet theo hang doc
    # if honCoor[1] - y >= 0:
    #     scale_y = real_size[1]/delta_y[0]
    # else:
    #     scale_y = real_size[1] / delta_y[1]
    # y_real = round((honCoor[1] - y)*scale_y, 2)
    # cv2.line(img, (x, y), (x, honCoor[1]), (0, 0, 255), 1)
    # cv2.putText(img, str(y_real) + 'cm', (x + 30, honCoor[1]), font, 1, (0, 255, 255))
    # Display Image was calculated the coordinates
    cv2.imshow("Calculated Image", img)
    return x_real, img
### Ham Tim Toa Do cua cac duong luoi
def GridCoordinates(image):
    # Input: Anh sau khi da tach khung
    # Output: Vecto toa do cua duong ke doc
    #          Vecto toa do cua duong ke ngang
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold the image
    ret, thres = cv2.threshold(gray, 127, 255, 0)

    # Step 1: Create an empty skeleton
    size = np.size(thres)
    skel = np.zeros(thres.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Repeat steps 2-4
    while True:
        # Step 2: Open the image
        open = cv2.morphologyEx(thres, cv2.MORPH_OPEN, element)
        # Step 3: Substract open from the original image
        temp = cv2.subtract(thres, open)
        # Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(thres, element)
        skel = cv2.bitwise_or(skel, temp)
        thres = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(thres) == 0:
            break
    # Displaying the final skeleton
    cv2.imshow("Skeleton Image", skel)
    # Find Coordinates of grid from bottom to top
    numVerDir = np.size(skel, 0) - 1
    verDir = skel[numVerDir, :]
    verCoor = DetermineCoordinate(verDir)
    while (True):
        if len(verCoor) > 0:
            if verCoor[0] <= 1:
                verCoor.pop(0)
        if len(verCoor) != numVer:
            numVerDir -= 1
            if numVerDir < 1:
                print(" Khong the tim dung duoc")
                break
            verDir = skel[numVerDir, :]
            verCoor = DetermineCoordinate(verDir)
        else:
            if min(abs(np.diff(verCoor))) < 20:
                numVerDir -= 1
                if numVerDir < 1:
                    print(" Khong the tim dung duoc")
                    break
                verDir = skel[numVerDir, :]
                verCoor = DetermineCoordinate(verDir)
            else:
                break
    print(" So duong doc: ", len(verCoor))
    print(verCoor)
    for i in range(len(verCoor)):
        cv2.circle(img, (verCoor[i], numVerDir), 1, (0, 0, 255), 1, cv2.LINE_AA)

    verCoorBottom = verCoor;
    # Find Coordinates of grid from top to bottom
    numVerDir = 0
    verDir = skel[numVerDir, :]
    verCoor = DetermineCoordinate(verDir)
    while (True):
        if len(verCoor) > 0:
            if verCoor[0] <= 1:
                verCoor.pop(0)
        if len(verCoor) != numVer:
            numVerDir += 1
            if numVerDir >= np.size(skel, 0):
                print(" Khong the tim dung duoc")
                break
            verDir = skel[numVerDir, :]
            verCoor = DetermineCoordinate(verDir)
        else:
            if min(abs(np.diff(verCoor))) < 20:
                numVerDir += 1
                if numVerDir >= np.size(skel, 0):
                    print(" Khong the tim dung duoc")
                    break
                verDir = skel[numVerDir, :]
                verCoor = DetermineCoordinate(verDir)
            else:
                break
    print(" So duong doc: ", len(verCoor))
    print(verCoor)
    for i in range(len(verCoor)):
        cv2.circle(img, (verCoor[i], numVerDir), 1, (0, 0, 255), 1, cv2.LINE_AA)

    verCoorTop = verCoor

    # numHonDir = int((verCoor[0] + verCoor[1]) / 2)
    # honDir = skel[:, numHonDir]
    #
    # honCoor = DetermineCoordinate(honDir)
    # i = 1
    # while (True):
    #     if len(honCoor) != honVer:
    #         numHonDir = int((verCoor[i] + verCoor[i + 1]) / 2)
    #         honDir = skel[:, numHonDir]
    #         honCoor = DetermineCoordinate(honDir)
    #         i += 1
    #         if i > len(verCoor):
    #             print("khong the tim dung dươc")
    #             break
    #     else:
    #         break
    #
    #
    # print(" So duong ngang: ", len(honCoor))
    # for i in range(len(honCoor)):
    #     cv2.circle(img, (numHonDir, honCoor[i]), 1, (0, 0, 255), 1, cv2.LINE_AA)

    for i in range(numVer):
        cv2.line(img, (verCoorTop[i], 0), (verCoorBottom[i], np.size(skel, 0)), (0, 0, 255), 1)

    for i in range(numVer):
        delta = abs(verCoorTop[i] - verCoorBottom[i])
        if delta > 10:
            print("Hinh bi cheo")
            break
    #Displaying the image with central points
    # cv2.imshow("Detected Coordinate", img)
    return verCoor
### Ham xac dinh toa do cua luoi
def DetermineCoordinate(dirValue):
    # Input: Vecto cua anh Skel theo mot huong nao do
    # Output: Vecto co chua vi tri cua duong luoi theo huong do
    i = 0
    coorValue = []
    while i < len(dirValue):
        if dirValue[i] != 0:
            valueSt = i
            i += 1
            while dirValue[i] != 0:
                i += 1
                continue
            valueNd = i - 1
            center = int((valueSt + valueNd) / 2)
            coorValue.append(center)
        else:
            i += 1
    return coorValue

## Ham xac dinh toa do trung tam cua laser Pointer
def FindCenter(image):
    # Input: Anh dau vao da tach khung
    # Output: Toa do (x, y) cua laser pointer
    img = image.copy()

    # whiteLower = (0, 0, 245)
    # whiteUpper = (255, 255, 255)

    blurredWar = cv2.GaussianBlur(img, (5, 5), 0)
    hsvWar = cv2.cvtColor(blurredWar, cv2.COLOR_BGR2HSV)

    maskWar = cv2.inRange(hsvWar, whiteLower, whiteUpper)
    # cv2.imshow("hsvWar1", maskWar)
    maskWar = cv2.dilate(maskWar, None, iterations=2)  # Đang là ảnh Gray với 2 mức xám 0 và 255
    maskWar = cv2.erode(maskWar, None, iterations=2)

    # cv2.imshow("hsvWar2", maskWar)
    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(maskWar, 127, 255, 0)
    # cv2.imshow("hsvWar3", thresh)
    # calculate moments of binary image
    # M = cv2.moments(thresh)
    #
    # # calculate x,y coordinate of center
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])
    # Focusing on [top right bottom left] of red region
    [top, bottom, left, right] = FindTRBL(thresh)
    x = int((right + left) / 2)
    y = int((top + bottom) / 2)

    # Improve the algorithm finding the centre laser
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # color -> gray
    delta = 20
    gray = gray[y - delta: y + delta, x - delta: x + delta]
    canny = cv2.Canny(gray, 50, 150, apertureSize=3)
    # dilate = cv2.dilate(canny, None, iterations=3)
    # erode = cv2.erode(dilate, None, iterations=4)
    cv2.imshow("Canny", canny)
    # cv2.imshow("dilate", dilate)
    # cv2.imshow("erode", erode)

    # Focusing on [top right bottom left] of red region
    [top, bottom, left, right] = FindTRBL(canny)
    cX = x + int((right + left)/2) - delta
    cY = y + int((top + bottom)/2) - delta

    return cX, cY