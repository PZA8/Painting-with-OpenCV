import cv2
import numpy as np
import time
import os
import handtrackingmodule as htm

#################

brushThickness = 15
eraserThickness = 50

#################

folderPath = "header"
myList = os.listdir(folderPath)
print(myList) #shows all images here (test here for fun)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))  #shows us that four images have been imported (test here why not)
header = overlayList[0]
drawColour = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 123)

detector = htm.handDetector(detectionCon=0.85, )
xp, yp = 0, 0


imgCanvas = np.zeros((720, 1280, 3), np.uint8)     #np.uint is unsigned which means that it can have 0 to 255 values
while True:
    #1.importing image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #2. Finding hand landmarks
    img = detector.findHands(img)   #test here
    lmList = detector.findPosition(img, draw=False)
    if len(lmList)!=0:
        #print(lmList)      #test here


        x1, y1 = lmList[8][1:]  #for tip of index fingers and we only unpack the landmark locations and ignore the index number
        x2, y2 = lmList[12][1:] #for tip of middle finger "" "" """ """

    #3. check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)

        #4. If selection mode - two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection mode")
            if y1 < 123:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColour = (255, 0, 255)     #number scheme for pink
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColour = (255, 0, 0)       #number scheme for blue since it is BGR so B is 255 and rest is off
                elif  800 < x1 < 950:
                    header = overlayList[2]
                    drawColour = (0, 255, 0)       #number scheme for green since it is BGR so G is 255 and rest is off
                elif 1050 < x1 < 1250:
                    header = overlayList[3]
                    drawColour = (0, 0, 0)         #number scheme for black
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColour,
                          cv2.FILLED)  # the addition bits give a dimensions to the rectangle between the two fingers
        #5. If drawing mode - index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1,y1), 15, drawColour, cv2.FILLED)  #no dimensions for a single finger square
            print("Drawing mode")
            if xp==0 and yp==0:      #is the first frame and we are starting to draw
                xp, yp = x1, y1      #first time it sees our finger it draws a line

            if drawColour == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColour, eraserThickness)  # this is for the eraser to be a little more thicker
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColour, eraserThickness)

            else:
                cv2.line(img, (xp,yp),(x1,y1), drawColour, brushThickness)     #this is used to draw a line
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColour, brushThickness)
            xp, yp = x1, y1            #here the points are updated once again so that the drawing is made and the points are constantly updated


    #use this part to improve the image blend
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)      #first get the grayscale image
    _, imgInv = cv2.threshold(imgGray, 25, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    #setting the header image
    img[0:123, 0:1280] = header
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("image", img)
    cv2.imshow("imagecanvas", imgCanvas)
    cv2.imshow("inv", imgInv)
    cv2.waitKey(1)
