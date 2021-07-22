import cv2
import numpy as np
import time
import autopy
import mediapipe as mp
import math
import wx
import threading
thread = None


# noinspection PyAttributeOutsideInit,PyShadowingNames
class TraceThread(threading.Thread):
    def __init__(self, *args, **keywords):
        threading.Thread.__init__(self, *args, **keywords)
        self.killed = False

    def start(self):
        self._run = self.run
        self.run = self.settrace_and_run
        threading.Thread.start(self)

    def settrace_and_run(self):
        import sys
        sys.settrace(self.globaltrace)
        self._run()

    def globaltrace(self, frame, event, arg):
        return self.localtrace if event == 'call' else None

    def localtrace(self, frame, event, arg):
        if self.killed and event == 'line':
            raise SystemExit()
        return self.localtrace


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        try:
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Fingers
            for id in range(1, 5):

                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        except IndexError:
            pass

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


class Mouse:

    def __init__(self):
        self.right_down = False
        self.left_down = False
        self.middle_down = False

    def stop_all(self):
        if self.left_down:
            autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)
            self.left_down = False

        if self.right_down:
            autopy.mouse.toggle(autopy.mouse.Button.RIGHT, False)
            self.right_down = False

        if self.middle_down:
            autopy.mouse.toggle(autopy.mouse.Button.MIDDLE, False)
            self.middle_down = False

    def left_click(self, command):
        autopy.mouse.toggle(autopy.mouse.Button.RIGHT, False)
        if command == 1:
            self.stop_all()
            autopy.mouse.toggle(autopy.mouse.Button.LEFT, True)
            self.left_down = True
            return

        if command == 3:
            autopy.mouse.toggle(autopy.mouse.Button.LEFT, True)
            self.left_down = True
            self.right_down = True

    def right_click(self, len1):
        if len1 >= 40:
            print("Right button down")
            self.stop_all()
            autopy.mouse.toggle(autopy.mouse.Button.RIGHT, True)
            self.right_down = True
            return

    def middle_click(self):
        autopy.mouse.toggle(autopy.mouse.Button.RIGHT, False)
        self.stop_all()
        if not self.middle_down:
            autopy.mouse.toggle(autopy.mouse.Button.MIDDLE, True)
            self.middle_down = True
        return

    def run(self, command, len1):
        if command == 0:
            self.stop_all()
        elif command == 1:
            if not self.left_down:
                self.left_click(command)
        elif command == 2:
            if not self.right_down:
                self.right_click(len1)
        elif command == 3:
            if not self.right_down:
                return
            self.left_click(command)

        elif command == 4:
            if not self.middle_down:
                self.middle_click()


def MouseCTRL(cam_on=False):
    wCam, hCam = 640, 480
    frameR = 100
    smoothening = 7
    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = handDetector(maxHands=1)
    wScr, hScr = autopy.screen.size()
    mouse = Mouse()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
        fingers = detector.fingersUp()
        num = 0
        for i in fingers:
            if i == 1:
                num += 1
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)
        try:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            mouse.run(num, length)
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                length, img, lineInfo = detector.findDistance(8, 12, img)
                if length < 40:
                    try:
                        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                        clocX = plocX + (x3 - plocX) / smoothening
                        clocY = plocY + (y3 - plocY) / smoothening
                        autopy.mouse.move(wScr - clocX, clocY * 1.26)
                        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        plocX, plocY = clocX, clocY
                    except Exception:
                        pass
        except IndexError:
            pass
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(num), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        if cam_on:
            cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    try:
        global thread
        thread.killed = True
    except Exception:
        pass


class User_Info(wx.Frame):

    def __init__(self, *args, **kw):
        super(User_Info, self).__init__(*args, **kw)
        self.InitUI()

    def InitUI(self):
        self.SetTitle('MouseCTRL')
        icon = wx.Icon()
        icon.CopyFromBitmap(wx.Bitmap("src.ico", wx.BITMAP_TYPE_ANY))
        self.SetIcon(icon)
        panel = wx.Panel(self)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        nm = wx.StaticBox(panel, -1, 'Settings')
        nm1 = wx.StaticBox(panel, -1, 'About Us')
        nmSizer = wx.StaticBoxSizer(nm, wx.VERTICAL)
        nm1Sizer = wx.StaticBoxSizer(nm1, wx.VERTICAL)
        nmbox = wx.BoxSizer(wx.VERTICAL)
        nm1box = wx.BoxSizer(wx.VERTICAL)

        font = wx.Font(16, family=wx.SCRIPT, weight=wx.BOLD, style=wx.ITALIC)
        st1 = wx.StaticText(panel, label="MouseCTRL", style=wx.ALIGN_LEFT)
        st1.SetFont(font)
        text = """MouseCTRL [version 1.2.2.7]
People actively involved in this projects were:

- Akarsh
- Aradhya
- Siddharth
__________________________________
© 2021: MouseCTRL & co.
        """
        self.newBtn = wx.Button(panel, wx.ID_ANY, 'Start MouseCTRL', size=(150, 30))
        self.newBtn1 = wx.Button(panel, wx.ID_ANY, 'Kill MouseCTRL process', size=(150, 30))
        self.newBtn2 = wx.Button(panel, wx.ID_ANY, 'How to Use MouseCTRL', size=(150, 30))
        self.nm4 = wx.TextCtrl(panel, -1, style=wx.TE_MULTILINE | wx.ALIGN_LEFT | wx.TE_NO_VSCROLL | wx.TE_READONLY,
                               size=(500, 150))
        self.nm4.AppendText(text)
        nmbox.Add(st1, 0, wx.ALL, 5)
        nmbox.Add(self.newBtn, 0, wx.ALL | wx.CENTER, 5)
        nmbox.Add(self.newBtn1, 0, wx.ALL | wx.CENTER, 5)
        nmbox.Add(self.newBtn2, 0, wx.ALL | wx.CENTER, 5)
        nm1box.Add(self.nm4, 0, wx.ALL | wx.CENTER, 5)
        nmSizer.Add(nmbox, 0, wx.ALL | wx.CENTER, 10)
        nm1Sizer.Add(nm1box, 0, wx.ALL | wx.CENTER, 10)
        hbox.Add(nmSizer, 0, wx.ALL, 5)
        hbox.Add(nm1Sizer, 0, wx.ALL, 5)
        self.newBtn.Bind(wx.EVT_BUTTON, self.runit)
        self.newBtn1.Bind(wx.EVT_BUTTON, self.runit2)
        panel.SetSizer(hbox)
        self.Centre()
        self.SetSize(wx.Size(450, 280))
        self.SetBackgroundColour('white')

    def runit(self, event):
        global thread
        thread = TraceThread(target=runmain)
        thread.run()

    def runit2(self, event):
        try:
            global thread
            thread.killed = True
        except Exception:
            pass
        wx.Exit()
        import sys
        sys.exit(0)


def runmain():
    MouseCTRL(True)


if __name__ == '__main__':
    app = wx.App()
    frame = User_Info(None, style=wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX))
    frame.Show()
    app.MainLoop()
