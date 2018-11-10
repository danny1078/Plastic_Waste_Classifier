import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)
firstFrame = None
min_area = 20000
trackers = {}
num_trackers = 0
tracker_ids_del = []
time_detected_init = None
time_detected_current = time_detected_init-time.time()
time_thres = 3000
detected = False


def create_tracker(image, x, y, w, h):
    global num_trackers
    if num_trackers <=5:
        trackers[num_trackers] = cv2.TrackerMedianFlow_create()
        trackers[num_trackers].init(image, (x, y, w, h))
        num_trackers += 1

def update_tracker(id, frame, gray_frame):
    global trackers
    state, bounding_box = trackers[id].update(frame)
    if state:
        (x, y, w, h) = bounding_box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        create_mask(gray_frame, x, y, w, h)
    else:
        cv2.putText(frame, 'tracking failure detected', (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        tracker_ids_del.append(id)
def delete_trackers():
    global tracker_ids_del
    global num_trackers
    print(tracker_ids_del)
    for id in tracker_ids_del:
        trackers.pop(id)
        num_trackers -= 1
    tracker_ids_del = []
def create_mask(gray_frame,    x, y, w, h):
    global firstFrame
    gray_frame[x-100:x+w+100,y-100:y+h+100] = firstFrame[x-100:x+w+100,y-100:y+h+100]

if not cap.isOpened():
    print("Error opening video stream")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    text = "No Motion Detected"
    if firstFrame is None:
        firstFrame = gray
    if num_trackers is not 0:
        for tracking_id in trackers:
            update_tracker(tracking_id, frame, gray)
        delete_trackers()
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        else:
            detected = True
            if time_detected_init is None:
                time_detected_init = time.time()
            else:
                if time_detected_current > time_thres:
                    (x,y,w,h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)
                    text = "Motion Detected"
                    create_tracker(frame,x, y, w, h)

    if not detected:
        time_detected_init = None
    cv2.putText(frame, 'time detected {}'.format(time_detected_current), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.imshow("Detection", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    print(num_trackers)
    key = cv2.waitKey(1) & 0xff
    if key == ord("q"):
        break
cv2.destroyAllWindows()
