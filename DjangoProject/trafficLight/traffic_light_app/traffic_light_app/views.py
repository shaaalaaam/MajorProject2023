from django.http import HttpResponse
from django.shortcuts import render
import cv2
import numpy as np
from fastapi import FastAPI
import traffic_light_app.constant as constant 
import threading
import os
import traffic_light_app.settings as settings


def compute_best_threshold(box_list, prev_box_list):
    """
    Computes the best overlap threshold between two sets of bounding boxes.
    
    Args:
    - box_list: list of lists of 4 integers, representing the (left, top, width, height) of the current bounding boxes
    - prev_box_list: list of lists of 4 integers, representing the (left, top, width, height) of the previous bounding boxes
    
    Returns:
    - best_threshold: float between 0 and 1, representing the best overlap threshold
    """
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]  # List of candidate thresholds
    best_count = 0
    best_threshold = 0

    for threshold in thresholds:
        count = 0

        for i in range(len(box_list)):
            for j in range(len(prev_box_list)):
                if box_overlap(box_list[i], prev_box_list[j]) > threshold:
                    count += 1

        if count > best_count:
            best_count = count
            best_threshold = threshold

    return best_threshold


def box_overlap(box, prev_box, threshold):
    """
    Computes the overlap (intersection over union) between two bounding boxes and returns True if the overlap is greater than the threshold.
    
    Args:
    - box: list of 4 integers, representing the (left, top, width, height) of the current bounding box
    - prev_box: list of 4 integers, representing the (left, top, width, height) of the previous bounding box
    - threshold: float between 0 and 1, representing the minimum overlap required to consider the boxes as the same object
    
    Returns:
    - overlap: bool, representing whether the overlap between the two bounding boxes is greater than the threshold or not
    """
    # Compute the coordinates of the intersection rectangle
    x1 = max(box[0], prev_box[0])
    y1 = max(box[1], prev_box[1])
    x2 = min(box[0] + box[2], prev_box[0] + prev_box[2])
    y2 = min(box[1] + box[3], prev_box[1] + prev_box[3])

    # Compute the area of intersection rectangle
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute the area of both the boxes
    box_area = box[2] * box[3]
    prev_box_area = prev_box[2] * prev_box[3]

    # Compute the area of union rectangle
    union_area = box_area + prev_box_area - intersection_area

    # Compute the overlap (intersection over union)
    overlap = intersection_area / union_area if union_area > 0 else 0

    # Check if the overlap is greater than the threshold
    return overlap > threshold


# def box_overlap(box, prev_box):
#     """
#     Computes the overlap (intersection over union) between two bounding boxes.
    
#     Args:
#     - box: list of 4 integers, representing the (left, top, width, height) of the current bounding box
#     - prev_box: list of 4 integers, representing the (left, top, width, height) of the previous bounding box
    
#     Returns:
#     - overlap: float between 0 and 1, representing the overlap between the two bounding boxes
#     """
#     # Compute the coordinates of the intersection rectangle
#     x1 = max(box[0], prev_box[0])
#     y1 = max(box[1], prev_box[1])
#     x2 = min(box[0] + box[2], prev_box[0] + prev_box[2])
#     y2 = min(box[1] + box[3], prev_box[1] + prev_box[3])

#     # Compute the area of intersection rectangle
#     intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

#     # Compute the area of both the boxes
#     box_area = box[2] * box[3]
#     prev_box_area = prev_box[2] * prev_box[3]

#     # Compute the area of union rectangle
#     union_area = box_area + prev_box_area - intersection_area

#     # Compute the overlap (intersection over union)
#     overlap = intersection_area / union_area if union_area > 0 else 0

#     return overlap

def index(request):
    # Call start_vehicle_detection function from within a view
    # print(request)
    start_vehicle_detection()
    vehicle_count=-3
    context = {'vehicle_count': vehicle_count}
    return render(request, "index.html", context)

def start_vehicle_detection():
    # #multithreading ka concept lagega yha
    # start_vehicle_detection0()
    # start_vehicle_detection1()
    # start_vehicle_detection2()
    # start_vehicle_detection3()
      # create four threads
    thread0 = threading.Thread(target=start_vehicle_detection0)
    thread1 = threading.Thread(target=start_vehicle_detection1)
    thread2 = threading.Thread(target=start_vehicle_detection2)
    thread3 = threading.Thread(target=start_vehicle_detection3)

    # start the threads
    thread0.start()
    thread1.start()
    thread2.start()
    thread3.start()

    # wait for all threads to complete
    thread0.join()
    thread1.join()
    thread2.join()
    thread3.join()
    
def start_vehicle_detection0():
    # vehicle_count = 0
    config_path0 = 'yolo/yolov3.cfg'
    config_full_path = os.path.join(settings.MEDIA_ROOT, config_path0)
    weight_path0 = 'yolo/yolov3.weights'
    weight_full_path = os.path.join(settings.MEDIA_ROOT, weight_path0)

    # Load the YOLOv3 model and initialize the video capture device
    # net = cv2.dnn.readNetFromDarknet(
    #     '/Users/shalam/Downloads/MajorProject/yolov3.cfg', '/Users/shalam/Downloads/MajorProject/yolov3.weights')
    net = cv2.dnn.readNetFromDarknet(
        config_full_path, weight_full_path)

    # Get the output layer names
    layer_names = net.getLayerNames()
    # print(layer_names)
    # for i in net.getUnconnectedOutLayers():
    #     print(i)
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    # Open video capture device
    
    video_path0 = 'videos/test0.mp4'
    video_full_path = os.path.join(settings.MEDIA_ROOT, video_path0)
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('/Users/shalam/Downloads/MajorProject/test0.mp4')
    print(video_full_path)
    cap = cv2.VideoCapture(video_full_path)

    detected_vehicles = []

    while True:
        vehicle_count = 0
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

                if confidence > 0.5 and class_id == 2:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    box = [x, y, width, height]
                    overlaps = False
                    for vehicle in detected_vehicles:
                        # threshold=compute_best_threshold(vehicle, box)
                        threshold=0.9
                        if box_overlap(vehicle, box,threshold):
                            overlaps = True
                            break

                    if not overlaps:
                        vehicle_count += 1
                        detected_vehicles.append(box)

                        boxes.append(box)
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

            cv2.putText(frame, label, (left, top-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(frame, label1, (left, top),
            label1 = f"Vehicle count: {vehicle_count}"
            cv2.putText(frame, label1, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            constant.VEHICLE_COUNT=vehicle_count
            # print(constant.VEHICLE_COUNT)
            # write a numeric value to a file
            with open('my_file0.txt', 'w') as f:
                f.write(str(vehicle_count))
        # Display the resulting frame
        # cv2.imshow('frame', frame)
        cv2.imwrite('lane_0.jpg', frame)
        print("Lane0 Captured")
        

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()
    # pass


def start_vehicle_detection1():
# vehicle_count = 0
    config_path0 = 'yolo/yolov3.cfg'
    config_full_path = os.path.join(settings.MEDIA_ROOT, config_path0)
    weight_path0 = 'yolo/yolov3.weights'
    weight_full_path = os.path.join(settings.MEDIA_ROOT, weight_path0)

    # Load the YOLOv3 model and initialize the video capture device
    # net = cv2.dnn.readNetFromDarknet(
    #     '/Users/shalam/Downloads/MajorProject/yolov3.cfg', '/Users/shalam/Downloads/MajorProject/yolov3.weights')
    net = cv2.dnn.readNetFromDarknet(
    config_full_path, weight_full_path)
    # Get the output layer names
    layer_names = net.getLayerNames()
    # print(layer_names)
    # for i in net.getUnconnectedOutLayers():
    #     print(i)
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    # Open video capture device
    video_path1 = 'videos/test1.mp4'
    video_full_path = os.path.join(settings.MEDIA_ROOT, video_path1)

    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('/Users/shalam/Downloads/MajorProject/test1.mp4')
    cap = cv2.VideoCapture(video_full_path)

    detected_vehicles = []

    while True:
        vehicle_count = 0
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

                if confidence > 0.5 and class_id == 2:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    box = [x, y, width, height]
                    overlaps = False
                    for vehicle in detected_vehicles:
                        # threshold=compute_best_threshold(vehicle, box)
                        threshold=0.9
                        if box_overlap(vehicle, box,threshold):
                            overlaps = True
                            break

                    if not overlaps:
                        vehicle_count += 1
                        detected_vehicles.append(box)

                        boxes.append(box)
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

            cv2.putText(frame, label, (left, top-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(frame, label1, (left, top),
            label1 = f"Vehicle count: {vehicle_count}"
            cv2.putText(frame, label1, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            constant.VEHICLE_COUNT=vehicle_count
            # print(constant.VEHICLE_COUNT)
            # write a numeric value to a file
            with open('my_file1.txt', 'w') as f:
                f.write(str(vehicle_count))
        # Display the resulting frame
        # cv2.imshow('frame', frame)
        cv2.imwrite('lane_1.jpg', frame)
        print("Lane1 Captured")

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()
    # pass

def start_vehicle_detection2():
    config_path0 = 'yolo/yolov3.cfg'
    config_full_path = os.path.join(settings.MEDIA_ROOT, config_path0)
    weight_path0 = 'yolo/yolov3.weights'
    weight_full_path = os.path.join(settings.MEDIA_ROOT, weight_path0)

    # Load the YOLOv3 model and initialize the video capture device
    # net = cv2.dnn.readNetFromDarknet(
    #     '/Users/shalam/Downloads/MajorProject/yolov3.cfg', '/Users/shalam/Downloads/MajorProject/yolov3.weights')
    net = cv2.dnn.readNetFromDarknet(
    config_full_path, weight_full_path)

    # Get the output layer names
    layer_names = net.getLayerNames()
    # print(layer_names)
    # for i in net.getUnconnectedOutLayers():
    #     print(i)
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    # Open video capture device
    video_path2 = 'videos/test2.mp4'
    video_full_path = os.path.join(settings.MEDIA_ROOT, video_path2)

    # cap = cv2.VideoCapture(0)
    # cap = cv2. VideoCapture('/Users/shalam/Downloads/MajorProject/test2.mp4')
    cap = cv2. VideoCapture(video_full_path)
    
    detected_vehicles = []

    while True:
        vehicle_count = 0
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

                if confidence > 0.5 and class_id == 2:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    box = [x, y, width, height]
                    overlaps = False
                    for vehicle in detected_vehicles:
                        # threshold=compute_best_threshold(vehicle, box)
                        threshold=0.9
                        if box_overlap(vehicle, box,threshold):
                            overlaps = True
                            break

                    if not overlaps:
                        vehicle_count += 1
                        detected_vehicles.append(box)

                        boxes.append(box)
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

            cv2.putText(frame, label, (left, top-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(frame, label1, (left, top),
            label1 = f"Vehicle count: {vehicle_count}"
            cv2.putText(frame, label1, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            constant.VEHICLE_COUNT=vehicle_count
            #print(constant.VEHICLE_COUNT)
            # write a numeric value to a file
            with open('my_file2.txt', 'w') as f:
                f.write(str(vehicle_count))
        # Display the resulting frame
        # cv2.imshow('frame', frame)
        cv2.imwrite('lane_2.jpg', frame)
        print("Lane2 Captured")

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()
    # pass

def start_vehicle_detection3():
    config_path0 = 'yolo/yolov3.cfg'
    config_full_path = os.path.join(settings.MEDIA_ROOT, config_path0)
    weight_path0 = 'yolo/yolov3.weights'
    weight_full_path = os.path.join(settings.MEDIA_ROOT, weight_path0)

    # Load the YOLOv3 model and initialize the video capture device
    # net = cv2.dnn.readNetFromDarknet(
    #     '/Users/shalam/Downloads/MajorProject/yolov3.cfg', '/Users/shalam/Downloads/MajorProject/yolov3.weights')
    net = cv2.dnn.readNetFromDarknet(
    config_full_path, weight_full_path)

    # Get the output layer names
    layer_names = net.getLayerNames()
    # print(layer_names)
    # for i in net.getUnconnectedOutLayers():
    #     print(i)
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    # Open video capture device
    video_path3 = 'videos/test3.mp4'
    video_full_path = os.path.join(settings.MEDIA_ROOT, video_path3)

    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('/Users/shalam/Downloads/MajorProject/test3.mp4')
    cap = cv2.VideoCapture(video_full_path)

    detected_vehicles = []

    while True:
        vehicle_count = 0
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

                if confidence > 0.5 and class_id == 2:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    box = [x, y, width, height]
                    overlaps = False
                    for vehicle in detected_vehicles:
                        # threshold=compute_best_threshold(vehicle, box)
                        threshold=0.9
                        if box_overlap(vehicle, box,threshold):
                            overlaps = True
                            break

                    if not overlaps:
                        vehicle_count += 1
                        detected_vehicles.append(box)

                        boxes.append(box)
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

            cv2.putText(frame, label, (left, top-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(frame, label1, (left, top),
            label1 = f"Vehicle count: {vehicle_count}"
            cv2.putText(frame, label1, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            constant.VEHICLE_COUNT=vehicle_count
            #print(constant.VEHICLE_COUNT)
            # write a numeric value to a file
            
        # Display the resulting frame
        # cv2.imshow('frame', frame)
        cv2.imwrite('lane_3.jpg', frame)
        print("Lane3 Captured")

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()
    # pass


