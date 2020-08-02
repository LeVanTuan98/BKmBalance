import numpy as np
import cv2
import imutils
from functions import *
blueLower = (90, 150, 0)
blueUpper = (120, 255, 255)

yellowLower = (20, 60, 60)
yellowUpper = (120, 255, 255)
# read image
image = cv2.imread("images/video27-6/MayTuan/frame0311.jpg")

orig = image
# STEP 1: Detect "Blue region"
blurred = cv2.GaussianBlur(image, (5, 5), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
maskBlue = cv2.inRange(hsv, blueLower, blueUpper)
maskBlue = cv2.erode(maskBlue, None, iterations=2)
maskBlue = cv2.dilate(maskBlue, None, iterations=2)
cv2.imshow("MaskBlue", maskBlue)

cntsBlue = cv2.findContours(maskBlue.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cntsBlue = imutils.grab_contours(cntsBlue)
cntsBlue = sorted(cntsBlue, key=cv2.contourArea, reverse=True)[:4]
for roiBlue in cntsBlue:
    peri = cv2.arcLength(roiBlue, True)
    approx = cv2.approxPolyDP(roiBlue, 0.02 * peri, True) # tao da giac voi do chinh xac xac dinh truoc

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
cv2.imshow("Result Filter Blue", warpedWhite)
blueFrame = cv2.resize(src=warpedWhite, dsize=(556, 356))
# STEP 1: Detect "Yellow region"
blurred = cv2.GaussianBlur(blueFrame, (5, 5), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
maskYellow = cv2.inRange(hsv, yellowLower, yellowUpper)
maskYellow = cv2.erode(maskYellow, None, iterations=2)
maskYellow = cv2.dilate(maskYellow, None, iterations=2)
cv2.imshow("MaskYellow", maskYellow)

cntsYellow = cv2.findContours(maskYellow.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cntsYellow = imutils.grab_contours(cntsYellow)
cntsYellow = sorted(cntsYellow, key=cv2.contourArea, reverse=True)[:4]
print(np.shape(cntsYellow))
for roiBlue in cntsYellow:
    peri = cv2.arcLength(roiBlue, True)
    approx = cv2.approxPolyDP(roiBlue, 0.02 * peri, True) # tao da giac voi do chinh xac xac dinh truoc

    maskBlue = np.zeros_like(blueFrame)
    cv2.drawContours(maskBlue, [approx], 0, (255, 255, 255), -1)  # Draw filled contour in mask
    warpedBlue = np.zeros_like(blueFrame)  # Extract out the object and place into output image
    warpedBlue[maskBlue == 255] = blueFrame[maskBlue == 255]

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
warpedWhite = four_point_transform(blueFrame, screenCntWhite.reshape(4, 2))
cv2.imshow("Result Filter Yellow", warpedWhite)



cv2.waitKey(0)
cv2.destroyAllWindows()
