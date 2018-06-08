import numpy as np
import cv2
import datetime


# some constants
BLUE = [255, 0, 0]  # rectangle color
RED = [0, 0, 255]  # PR BG
GREEN = [0, 255, 0]  # PR FG
BLACK = [0, 0, 0]  # sure BG
WHITE = [255, 255, 255]  # sure FG

DRAW_BG = {'color': BLACK, 'val': 0}
DRAW_FG = {'color': WHITE, 'val': 1}
DRAW_PR_FG = {'color': GREEN, 'val': 3}
DRAW_PR_BG = {'color': RED, 'val': 2}



class SegmenterControl:
    def __init__(self):
        self.image = None
        self.image2 = None
        self.mask = None
        self.output = None
        self.drawing = False
        self.initialized = False
        self.value = DRAW_FG
        self.coords = []
        self.lastCenter = None
        self.lastRadius = -1
        self.models = [np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)]

    def setImage(self, newImage):
        print('set image')
        self.image = newImage
        self.image2 = newImage.copy()
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8) + DRAW_BG['val']
        self.output = newImage.copy()
        self.initialized = False
        self.coords = []
        self.models = [np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)]

    def getMask(self):
        return self.mask

    def getImage(self):
        return self.image

    def getOutput(self):
        return self.output

    def getLastCenter(self):
        return self.lastCenter

    def startDrawing(self):
        self.drawing = True

    def stopDrawing(self):
        self.drawing = False

    def switchTo(self, newValue):
        self.value = newValue

    def drawCircle(self, x, y, radius=30):
        if self.drawing:
            cv2.circle(self.image, (x, y), radius, self.value['color'], -1)
            cv2.circle(self.mask, (x, y), radius, self.value['val'], -1)
            self.coords.append((y, x))
            # print((x, y))

    def drawDoubleRing(self, x, y):
        # if self.drawing:
        overlay = self.image.copy()
        cv2.circle(overlay, (x, y), 150, DRAW_PR_BG['color'], -1)
        cv2.circle(overlay, (x, y), 150, DRAW_BG['color'], 50)
        cv2.circle(overlay, (x, y), 50, DRAW_FG['color'], -1)
        cv2.addWeighted(overlay, 0.3, self.image, 1 - 0.3, 0, self.image)

        cv2.circle(self.mask, (x, y), 150, DRAW_PR_BG['val'], -1)
        cv2.circle(self.mask, (x, y), 150, DRAW_BG['val'], 50)
        cv2.circle(self.mask, (x, y), 50, DRAW_FG['val'], -1)
        self.coords.append((y - 150, x - 150))
        self.coords.append((y + 150, x + 150))

    def findLargestContour(self, contours):
        maxArea = 0
        largestContour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > maxArea:
                largestContour = cnt
                maxArea = area

        return largestContour, maxArea

    def segment(self):

        #mask2 = self.mask.copy()
        #cv2.imshow('mask2 2', self.mask * 64)
        # self.initialized = True
        # if not self.initialized:
        #     print('initialize')
        #     npCoords = np.array(self.coords)
        #     minCoord = np.amin(npCoords, 0)
        #     maxCoord = np.amax(npCoords, 0)
        #     rect = (minCoord[0], minCoord[1], maxCoord[0] - minCoord[0], maxCoord[1] - minCoord[1])
        #     cv2.grabCut(self.image2, mask2, rect, self.models[0], self.models[1], 0, cv2.GC_INIT_WITH_RECT)
        #     # cv2.imshow('mask2', mask2 * 64)
        #     mask2[self.mask == 0] = 0
        #     mask2[self.mask == 1] = 1
        #     self.initialized = True
        #
        # # cv2.imshow('mask2 original', mask2 * 64)
        #mask2[self.mask == 0] = 0
        #cv2.imshow('mask2 2', mask2 * 64)
        cv2.grabCut(self.image2, self.mask, None, self.models[0], self.models[1], 1, cv2.GC_INIT_WITH_MASK)
        # cv2.imshow('mask2 3', mask2 * 64)

        # mask2 = np.where((self.mask == 1) | (self.mask == 3), 1, 0).astype('uint8')
        # self.output = cv2.bitwise_and(self.output, self.output, mask=mask2)
        #self.mask = mask2.copy()
        mask2 = np.where((self.mask == 1) + (self.mask == 3), 1, 0).astype('uint8')
        # np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')

        # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9)))

        self.output = self.image2 * mask2[:, :, np.newaxis]
        # cv2.imshow('foo', mask2)
        print('finished')

        self.output2 = self.output.copy()

        cnts = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        cnt, area = self.findLargestContour(cnts)
        #
        mask3 = np.zeros(self.image.shape[:2], dtype=np.uint8)
        mask3 = cv2.cvtColor(mask3 * 255, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(mask3, [cnt], -1, (255, 255, 255), -1)
        cv2.addWeighted(mask3, 0.3, self.image2, 1 - 0.3, 0, self.output2)
        #
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        self.lastCenter = (cX, cY)
        cv2.circle(self.output2, (cX, cY), 2, DRAW_FG['color'], 2)
        #print(cX, cY)
        #
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(self.output2, center, radius, (255, 255, 255), 2)

        if self.lastRadius is -1:
            self.lastRadius = radius

        radiusUp = self.lastRadius + (self.lastRadius * 0.1)
        radiusDown = self.lastRadius - (self.lastRadius * 0.1)
        if radiusDown < radius < radiusUp:
            self.lastRadius = radius
        else:
            print('something\'s strange')


    def run(self):
        start_time = datetime.now()
        self.segment()
        end_time = datetime.now()
        print('Duration: {0}'.format(end_time - start_time))

        cv2.imshow('overlay', self.output2)
