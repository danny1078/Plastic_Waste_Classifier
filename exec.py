import cv2

cap = cv2.VideoCapture(0)
firstFrame = None
min_area = 300
trackers = {}
num_trackers = 0

def create_tracker(image, x, y, w, h):
    global num_trackers
    trackers[num_trackers] = cv2.TrackerKCF_create()
    trackers[num_trackers].init(image, (x, y, w, h))
    num_trackers += 1

def update_draw_tracker(id, image):
    global trackers
    state, bounding_box = trackers[id].update(image)
    if state:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        (x, y, w, h) = bounding_box
    else:
        cv2.putText(image, 'tracking failure detected', (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        trackers.pop(id)

def create_mask()
if not cap.isOpened():
    print("Error opening video stream")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    text = "No Motion Detected"
    if firstFrame is None:
        firstFrame = gray
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)
        text = "Motion Detected"
        create_tracker(frame,x, y, w, h)
    for tracking_id in trackers:
        update_draw_tracker(tracking_id, frame)

    cv2.imshow("Detection", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)

    key = cv2.waitKey(1) & 0xff
    if key == ord("q"):
        break
cv2.destroyAllWindows()
