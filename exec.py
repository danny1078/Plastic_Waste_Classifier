import cv2
import time
from keras.models import load_model
import numpy as np
import serial

cap = cv2.VideoCapture(0)
firstFrame = None
min_area = 2000
trackers = {}
num_trackers = 0
tracker_ids_del = []
time_detected_init = None
time_thres = 3
detected = False
model = load_model('/Users/Danny Han/Desktop/Plastic_Waste_Identifier_Project/Models/Xception_Final.h5')
classes = ['HDPE', 'LDPE', 'PET', 'PP', 'PS', 'PVC']
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=500, backgroundRatio=0.025)
#arduino = serial.Serial('COM3', 9600)


def classify_object(imarr):
    imarr = np.expand_dims(imarr, axis=0)/255.
    pred = model.predict_classes(imarr)
    obj_type = classes[pred[0]]

    return obj_type

def create_tracker(image, x, y, w, h):
    global num_trackers
    global time_detected_init
    if num_trackers <=5:
        trackers[num_trackers] = []
        trackers[num_trackers].append(cv2.TrackerMedianFlow_create())
        trackers[num_trackers][0].init(image, (x, y, w, h))
        cropped_img = image[x:x + w, y:y + h]
        resized_cropped_img = cv2.resize(cropped_img,(400,400))
        obj_type = classify_object(resized_cropped_img)
        trackers[num_trackers].append(obj_type)
        trackers[num_trackers].append((x, y, w, h))
        num_trackers += 1
        time_detected_init = None


def update_tracker(id, frame, gray_frame):
    global trackers
    state, bounding_box = trackers[id][0].update(frame)
    if state:
        (x, y, w, h) = bounding_box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        trackers[id][2] = (x, y, w, h)
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
def create_mask(gray_frame,y, x, h, w):
    global firstFrame
    gray_frame[x:x+w,y:y+h] = firstFrame[x:x+w,y:y+h]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255 ,0), 2)

if not cap.isOpened():
    print("Error opening video stream")

start = time.time()
while time.time() - start < 2:
    continue

while True:
    #arduino.write(('1'.encode("utf-8")))
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    text = "No Motion Detected"
    if firstFrame is None:
        firstFrame = gray
    if num_trackers is not 0:
        for tracking_id in trackers:
            update_tracker(tracking_id, frame, gray)
        delete_trackers()
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]
    #fgmask = fgbg.apply(gray)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]
    if len(cnts) != 0:
        c = max(cnts, key=cv2.contourArea)

        if cv2.contourArea(c) > min_area:
             detected = True
             if time_detected_init is None:
                time_detected_init = time.time()
             else:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'detected', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
             if time.time()- time_detected_init > time_thres:
                create_tracker(frame, x, y, w, h)

    if not detected:
        time_detected_init = None
    detected = False
    if time_detected_init is not None:
        cv2.putText(frame, 'time detected {}'.format(time.time() - time_detected_init), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    for id in trackers:
        (x, y, w, h) = trackers[id][2]
        cv2.putText(frame,'Object type = {}'.format(trackers[id][1]), (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2 )
    cv2.imshow("Detection", frame)
    cv2.imshow("thresh", thresh)
    cv2.imshow("delta", frameDelta)
    print(num_trackers)
    key = cv2.waitKey(1) & 0xff

    if key == ord("q"):
        break
cv2.destroyAllWindows()
