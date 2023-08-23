import cv2
import numpy as np

config_path = 'yolov4-tiny-obj.cfg'
weights_path = 'yolov4-tiny-obj_last.weights'

class_labels = ['bottle', 'cup','pencil']
class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

net = cv2.dnn.readNet(config_path, weights_path)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(obj[0] * frame.shape[1])
                center_y = int(obj[1] * frame.shape[0])
                width = int(obj[2] * frame.shape[1])
                height = int(obj[3] * frame.shape[0])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                label = f"{class_labels[class_id]}: {confidence:.2f}"
                color = class_colors[class_id]
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('YOLO Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
