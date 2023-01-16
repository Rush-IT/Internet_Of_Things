

from turtle import color
import cv2
import numpy as np
import time

capture = cv2.VideoCapture(0)

classes = ["background", "person", "bicycle", "car", "motorcycle",
  "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
  "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
  "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
  "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
  "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
  "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
  "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
  "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

pb = 'frozen_inference_graph.pb'
pbtxt = 'ssd_inception_v2_coco_2017_11_17.pbtxt'

cvNet = cv2.dnn.readNetFromTensorflow(pb, pbtxt)

finish = 0


while True:
    ret, img = capture.read()
    start = time.time()
    rows = img.shape[0]
    colums = img.shape[1]
    if start - finish > 1:
        cvNet.setInput(cv2.dnn.blobFromImage(img, size = (300, 300), swapRB = True, crop = False))
        finish = time.time()
    cvOut = cvNet.forward()
    print(cvOut)
    for det in cvOut[0, 0, :, :]:
        score = float(det[2])
        if score > 0.5:
            
            idx = int(det[1])
            left = det[3] * colums
            top = det[4] * rows
            right = det[5] * colums
            bottom = det[6] * rows
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 0), thickness=1)
            cv2.putText(img, classes[idx], (int(left), int(top)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)


    cv2.imshow('FromCamera', img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()
