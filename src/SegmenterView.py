import SegmenterControl
import cv2



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


class SegmenterView:
    def __init__(self):
        self.windowName = 'Ultrasound image'
        self.windowNameMask = 'Mask image'
        self.windowNameOutput = 'Output image'
        self.next = False
        self.doDoubleRing = False
        self.lastCoords = None
        cv2.namedWindow(self.windowName)
        cv2.setMouseCallback(self.windowName, self.onmouse)
        cv2.namedWindow(self.windowNameMask)
        cv2.namedWindow(self.windowNameOutput)
        self.segmenter = SegmenterControl()
        pass

    def onmouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.lastCoords = (x, y)
            if not self.doDoubleRing:
                self.segmenter.startDrawing()
                self.segmenter.drawCircle(x, y)
            else:
                self.segmenter.drawDoubleRing(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.segmenter.drawCircle(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if not self.doDoubleRing:
                self.segmenter.stopDrawing()
                self.segmenter.drawCircle(x, y)
            else:
                self.segmenter.drawDoubleRing(x, y)

        self.updateView()

    def updateView(self):
        cv2.imshow(self.windowName, self.segmenter.getImage())
        mask = self.segmenter.getMask()
        #mask = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        cv2.imshow(self.windowNameMask, mask * 64)
        cv2.imshow(self.windowNameOutput, self.segmenter.getOutput())

    def handleKey(self, key):
        k = key
        # key bindings
        if k == 27:  # esc to exit
            exit(0)
        elif k == ord('0') or k == ord('§'):  # BG drawing
            print(" mark background regions with left mouse button \n")
            self.segmenter.switchTo(DRAW_BG)
            self.doDoubleRing = False
        elif k == ord('1'):  # FG drawing
            print(" mark foreground regions with left mouse button \n")
            self.segmenter.switchTo(DRAW_FG)
            self.doDoubleRing = False
        elif k == ord('2'):  # PR_BG drawing
            self.segmenter.switchTo(DRAW_PR_BG)
            self.doDoubleRing = False
        elif k == ord('3'):  # PR_FG drawing
            self.segmenter.switchTo(DRAW_PR_FG)
            self.doDoubleRing = False
        elif k == ord('4'):
            print('using fancy double ring')
            self.doDoubleRing = True
        elif k == ord('n') or k == ord(' '):
            self.next = True
        elif k == ord('e'):
            self.segmenter.run()
        elif k == ord('r'):
            #self.segmenter.drawDoubleRing(self.lastCoords[0], self.lastCoords[1])
            self.segmenter.drawDoubleRing(self.segmenter.getLastCenter()[0], self.segmenter.getLastCenter()[1])
        elif k == ord('m'):
            self.segmenter.drawDoubleRing(self.segmenter.getLastCenter()[0], self.segmenter.getLastCenter()[1])
            self.segmenter.run()

    def show(self, image):
        self.segmenter.setImage(image)
        self.next = False
        while not self.next:
            self.updateView()
            k = cv2.waitKey(0)
            self.handleKey(k)

