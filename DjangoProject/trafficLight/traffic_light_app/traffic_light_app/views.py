from django.http import HttpResponse
from django.shortcuts import render
import cv2
import numpy as np


def index(request):
    # Call start_vehicle_detection function from within a view
    vehicle_count = start_vehicle_detection()
    context = {'vehicle_count': vehicle_count}
    return render(request, "indexx.html", context)

    

def start_vehicle_detection():
    # vehicle_count = 0
    # Load the YOLOv3 model and initialize the video capture device
    net = cv2.dnn.readNetFromDarknet(
        '/Users/shalam/Downloads/MajorProject/yolov3.cfg', '/Users/shalam/Downloads/MajorProject/yolov3.weights')


    # Get the output layer names
    layer_names = net.getLayerNames()
    # print(layer_names)
    # for i in net.getUnconnectedOutLayers():
    #     print(i)
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    # Open video capture device
    cap = cv2.VideoCapture(0)

    while True:
        vehicle_count=0
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert frame to blob for YOLO model input
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (416, 416), swapRB=True, crop=False)

        # Set input to the network
        net.setInput(blob)

        # Perform forward pass through the network
        layer_outputs = net.forward(output_layers)

        # Process the outputs of the YOLOv3 model
        class_ids = []
        confidences = []
        boxes = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and class_id==2:
                    vehicle_count+=1
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Draw bounding boxes around detected objects
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        for i in indices:
            i = i
            box = boxes[i]
            left, top, width, height = box
            cv2.rectangle(frame, (left, top),
                        (left+width, top+height), (0, 255, 0), 2)
            label = f"{class_ids[i]}: {confidences[i]:.2f}"
            
            cv2.putText(frame , label, (left, top-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(frame, label1, (left, top),
            label1 = f"Vehicle count: {vehicle_count}"
            cv2.putText(frame, label1, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Display the resulting frame
        cv2.imshow('frame', frame)
        

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()


# Start the vehicle detection loop when the Django app starts
start_vehicle_detection()
