import cv2

# Load YOLOv3
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Process Video
cap = cv2.VideoCapture('bangkok.mp4')

while True:
    _, frame = cap.read()
    height, width, channels = frame.shape

    # Detect Objects (Kendaraan)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []  # Untuk menyimpan ID kelas yang terdeteksi
    confidences = []  # Untuk menyimpan tingkat kepercayaan deteksi
    boxes = []  # Untuk menyimpan koordinat bounding boxes

    # Parse Output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 2:  # Kendaraan (sesuaikan dengan kelas yang diinginkan)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Koordinat pojok kotak
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

                # Gambar kotak di sekitar kendaraan
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Gunakan non-maximum suppression untuk menghilangkan deteksi yang tumpang tindih
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 5), font, 1, (0, 255, 0), 1)

    # Tampilkan video yang telah diolah
    cv2.imshow('Detected Vehicles', frame)
    key = cv2.waitKey(1)
    if key == 27:  # Tekan 'Esc' untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
