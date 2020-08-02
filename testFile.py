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


blackLower = (0, 0, 60)
blackUpper = (255, 150, 120)

redLower = (0, 0, 0)
redUpper = (40, 255, 255)

yellowLower = (100, 100, 100)
yellowUpper = (120, 255, 255)

blueLower = (90, 150, 0)
blueUpper = (120, 255, 255)

# read image
image = cv2.imread("images/video27-6/MayTuan/frame0311.jpg")

# resize image
orig = image
# ratio = image.shape[0] / 500.0  # Chiều cao ảnh chuẩn hóa (chia cho 500)
# image = imutils.resize(image, height=500)

# convert the image to grayscale, blur it, and find edges
# in the image
blurred = cv2.GaussianBlur(image, (5, 5), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
cv2.imshow("blurred", blurred)
maskBlue = cv2.inRange(hsv, blueLower, blueUpper)
cv2.imshow("MaskBlue1", maskBlue)
maskBlue = cv2.erode(maskBlue, None, iterations=2)
cv2.imshow("MaskBlue2", maskBlue)
maskBlue = cv2.dilate(maskBlue, None, iterations=2)

cv2.imshow("MaskBlue3", maskBlue)

# show the original image and the edge detected image
print("STEP 1: Color Detection")
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
    warpedBlue = cv2.medianBlur(warpedBlue, 5)
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
            screenCntWhite = approx
            break

warpedWhite = four_point_transform(orig, screenCntWhite.reshape(4, 2))

print("STEP 2: Rounding blue paper")
cv2.imshow("Outline", warpedWhite)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# for c in cnts:
#     # approximate the contour
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#     # if our approximated contour has four points, then we
#     # can assume that we have found our screen
#     if len(approx) == 4:
#         screenCntBlue = approx
#         break
#
# if len(approx) != 4:
#     print("Không thấy tờ giấy màu xanh trong frame thứ")
#
# # show the contour (outline) of the piece of paper
# print("STEP 2: Rounding blue paper")
# cv2.drawContours(image, [screenCntBlue], -1, (0, 255, 0), 2)
# cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# Tách cùng có màu xanh lá cây
warpedBlue = four_point_transform(orig, screenCntWhite.reshape(4, 2))
origBlue = warpedBlue
warpedBlue = cv2.medianBlur(warpedBlue, 5)
gray = cv2.cvtColor(warpedBlue, cv2.COLOR_BGR2GRAY)

binnary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 3, 2)

# gray = cv2.cvtColor(warpedBlue, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(binnary, 10, 200)

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCntBlack = approx
        break

# show the contour (outline) of the piece of paper
print("STEP 3: Tách vùng quan tâm")
cv2.drawContours(warpedBlue, [screenCntBlack], -1, (0, 255, 0), 2)
warpedContours = four_point_transform(origBlue, screenCntBlack.reshape(4, 2))
cv2.imshow("Outline", binnary)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Step 3: Apply a Perspective Transform & Threshold
# # apply the four point transform to obtain a top-down
# # view of the original image
# try:
#     warpedWhite = four_point_transform(warpedGreen, screenCntWhite.reshape(4, 2) * ratio)
#     # convert the warped image to grayscale, then threshold it
#     # to give it that 'black and white' paper effect
#     # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#     # T = threshold_local(warped, 11, offset=10, method="gaussian")
#     # warped = (warped > T).astype("uint8") * 255
#     # show the original and scanned images
#     fileNameImage = 'anhOut%s.jpg' % (frameThu)
#     print(fileNameImage)
#     cv2.imwrite(fileNameImage, warpedGreen)
# except:
#     print("Lỗi ở frame thứ", frameThu)
#
# # Hiện thị video
# cv2.destroyAllWindows()