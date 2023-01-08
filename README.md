# Vehicle Counting Application using YOLOv3

## YOLO (You Only Look Once)

You Only Look Once (YOLO) is a CNN architecture for performing real-time object detection. The algorithm applies a single neural network to the full image, and then divides the image into regions and predicts bounding boxes and probabilities for each region

### Requirements for running the project

- python (>=3.6)

- openCV (Stable version)

- imutils

- sciPY

- numPY

- weights file for the coco dataset

```
https://pjreddie.com/media/files/yolov3.weights
```

### Runnning the Application

```
python yolo_video.py --input inputVideos/highway.mp4 --output outputVideos/highwayOut.avi --yolo yolo-coco
```

```
python yolo_video.py --input inputVideos/highway.mp4 --output outputVideos/highwayOut.avi --yolo yolo-coco --confidence 0.3
```

### Resources

- https://medium.com/analytics-vidhya/object-detection-using-yolov3-d48100de2ebb

- https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html

- https://www.analyticsvidhya.com/blog/2021/06/implementation-of-yolov3-simplified/
