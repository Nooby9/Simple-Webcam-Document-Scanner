import cv2
import numpy as np

widthImg = 816
heightImg = 1054

frameWidth = 1920
frameHeight = 1080
cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
#cap.set(10, 150) #brightness of image
def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts to grayScale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)

    #making the edge thicker:
    kernel = np.ones((5, 5)) #a matrix of 1s
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)

    return imgThres

def getCopntours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3) #draw contour lines on the image
            peri = cv2.arcLength(cnt, True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True) # find approximation of corner points
            if area > maxArea and len(approx) == 4: #four sides of paper
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)  # draw contour lines on the image
    return biggest #returns the biggest contour

def reorder(myPoints): #reorders the points in the matrix such that the smallest is always in front and largest in back

    myPoints = myPoints.reshape((4, 2)) #.shape return (4, 1, 2), so change it into (4,2): 4 points, x, y
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)] #assign the smallest sum of coordinates to the first point
    myPointsNew[3] = myPoints[np.argmax(add)] #assign the largest sum of coordinates to the last point
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)] #assign the smallest difference of coordinates to the second point
    myPointsNew[2] = myPoints[np.argmax(diff)] #assign the largest difference of coordinates to the third point
    return myPointsNew

def getWarp(img, biggest):

    biggest = reorder(biggest)
    #print(biggest.shape)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]) #requires reordering
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutPut = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    #crop the boundry pixels so theres no black bars
    imgCropped = imgOutPut[20:imgOutPut.shape[0]-20, 20:imgOutPut.shape[1]-20]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))

    return imgCropped


while True:
    success, img = cap.read()
    cv2.resize(img,(widthImg, heightImg))
    imgContour = img.copy()
    imgThres = preProcessing(img)
    biggest = getCopntours(imgThres)
    if biggest.size != 0:
        imgWarped = getWarp(img, biggest)
        cv2.imshow("Result", imgWarped)

    cv2.imshow("Window", imgContour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
