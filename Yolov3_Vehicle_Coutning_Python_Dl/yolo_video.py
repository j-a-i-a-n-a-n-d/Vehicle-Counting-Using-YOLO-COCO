import numpy as np
import imutils
import time
from scipy import spatial
import cv2
from input_retrieval import *

list_of_vehicles = ["bicycle", "car", "motorbike", "bus", "truck", "train"]
FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416

# Parse command line arguments and extract the values required
LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath, preDefinedConfidence, preDefinedThreshold = parseCommandLineArguments()
# Initialize a list of colors to represent each possible class label
np.random.seed(42)  # legacy
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


# display the Vehicle Count on top left corner
def displayVehicleCount(frame, vehicle_count):
    cv2.putText(img=frame, text='Detected Vehicles: ' + str(vehicle_count), org=(20, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0xFF, 0), thickness=2)


# Displaying the FPS of the detected video
def displayFPS(start_time, num_frames, frame_no):
    current_time = int(time.time())
    if (current_time > start_time):
        os.system('cls')
        print("FPS:", num_frames)
        frame_no += num_frames
        print("F NUMBER :", frame_no)
        num_frames = 0
        start_time = current_time
    return start_time, num_frames, frame_no


# Draw all the detection boxes with a green dot at the center
# step after non maxium supression
def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
    if len(idxs) > 0:
        print(idxs)
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            # draw a bounding box rectangle and label on the frame
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(
                x + w, y + h), color=color, thickness=2)
            text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(img=frame, text=text, org=(x, y - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=2)
            # Draw a green dot in the middle of the box
            cv2.circle(img=frame, center=(x + (w//2), y + (h//2)),
                       radius=1, color=(0, 0xFF, 0), thickness=2)


# Initializing the video writer with the output video path and the same number of fps, width and height as the source video
def initializeVideoWriter(video_width, video_height, videoStream):
    # Getting the fps of the source video
    sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
    # initialize our video writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps, (video_width, video_height), True)


# PURPOSE: Identifying if the current box was present in the previous frames

def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
    centerX, centerY, width, height = current_box
    dist = np.inf  # (Initializing the minimum distance infinity)
    # Iterating through all the k-dimensional trees
    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list = list(previous_frame_detections[i].keys())
        if len(coordinate_list) == 0:  # When there are no detections in the previous frame
            continue
        # Finding the distance to the closest point and the index
        temp_dist, index = spatial.KDTree(
            coordinate_list).query([(centerX, centerY)])
        if (temp_dist < dist):
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]

    if (dist > (max(width, height)/2)):
        return False

    # Keeping the vehicle ID constant
    current_detections[(centerX, centerY)
                       ] = previous_frame_detections[frame_num][coord]
    return True


def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
    current_detections = {}
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            centerX = x + (w//2)
            centerY = y + (h//2)

            # When the detection is in the list of vehicles, AND
            # it crosses the line AND
            # the ID of the detection is not present in the vehicles
            if (LABELS[classIDs[i]] in list_of_vehicles):
                current_detections[(centerX, centerY)] = vehicle_count
                if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)):
                    vehicle_count += 1
                    # vehicle_crossed_line_flag += True
                # else: #ID assigning
                    # Add the current detection mid-point of box to the list of detected items
                # Get the ID corresponding to the current detection

                ID = current_detections.get((centerX, centerY))
                # If there are two detections having the same ID due to being too close,
                # then assign a new ID to current detection.
                if (list(current_detections.values()).count(ID) > 1):
                    current_detections[(centerX, centerY)] = vehicle_count
                    vehicle_count += 1

                # Display the ID at the center of the box
                cv2.putText(frame, str(ID), (centerX, centerY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

    return vehicle_count, current_detections


# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the output layer names that we need from YOLO
print("Loading YOLO from disk...")
print("=========================")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]


# frame dimensions
videoStream = cv2.VideoCapture(inputVideoPath)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialization
previous_frame_detections = [{(0, 0): 0} for i in range(FRAMES_BEFORE_CURRENT)]
num_frames, vehicle_count = 0, 1
writer = initializeVideoWriter(video_width, video_height, videoStream)
start_time = int(time.time())
global frame_no
frame_no = 0

# loop over video
while True:
    print("================NEW FRAME================")
    num_frames += 1
    print("FRAME:\t", num_frames)
    # Initialization for each iteration
    boxes, confidences, classIDs = [], [], []
    # Calculating fps each second
    start_time, num_frames, frame_no = displayFPS(
        start_time, num_frames, frame_no)
    # read the next frame from the file
    (grabbed, frame) = videoStream.read()
    if not grabbed:
        break

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (inputWidth, inputHeight), swapRB=True, crop=False)  # 4d array
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for i, detection in enumerate(output):
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > preDefinedConfidence:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * \
                    np.array([video_width, video_height,
                              video_width, video_height])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Printing the info of the detection
                print('\nName:\t', LABELS[classID], '\t|\tBOX:\t', x, y)

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # # Changing line color to green if a vehicle in the frame has crossed the line
    # if vehicle_crossed_line_flag:
    # 	cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0xFF, 0), 2)
    # # Changing line color to red if a vehicle in the frame has not crossed the line
    # else:
    # 	cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0, 0xFF), 2)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(
        boxes, confidences, preDefinedConfidence, preDefinedThreshold)
    # Draw detection box
    drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

    vehicle_count, current_detections = count_vehicles(
        idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame)

    # Display Vehicle Count if a vehicle has passed the line
    displayVehicleCount(frame, vehicle_count)

    # write the output frame to disk
    writer.write(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Updating with the current frame detections
    previous_frame_detections.pop(0)  # Removing the first frame from the list
    # previous_frame_detections.append(spatial.KDTree(current_detections))
    previous_frame_detections.append(current_detections)

# release the file pointers
print("INFO cleaning up stopping")
print("=========================")
writer.release()
videoStream.release()
