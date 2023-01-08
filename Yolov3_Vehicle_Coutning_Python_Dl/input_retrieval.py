import argparse
import os

# parsing the command line arguments


def parseCommandLineArguments():
    # parsing the arguments
    ap = argparse.ArgumentParser()  # argparse object
    ap.add_argument("-i", "--input", required=True,
                    help="path to i/p video")
    ap.add_argument("-o", "--output", required=True,
                    help="path to o/p video")
    ap.add_argument("-y", "--yolo", required=True,
                    help="YOLO dir")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="threshold when applying non-maxima suppression")

    args = vars(ap.parse_args())
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    inputVideoPath = args["input"]
    outputVideoPath = args["output"]
    confidence = args["confidence"]
    threshold = args["threshold"]
    return LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath, confidence, threshold
    #    coco.names,yolov3.weights,yolov3.cfg,i/p video path,o/p video path, confidence(accuracy), nms(threshold)

    # python yolo_video.py --input inputVideos/highway.mp4 --output outputVideos/highwayOut.avi --yolo yolo-coco --confidence 0.3
    # python yolo_video.py -i inputVideos/highway.mp4 -o outputVideos/highwayOut.avi -y yolo-coco -c 0.7 -t 0.7
    # python yolo_video.py --input inputVideos/highway.mp4 --output outputVideos/highwayOut.avi --yolo yolo-coco
