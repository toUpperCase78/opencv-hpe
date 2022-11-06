import cv2
import time
import numpy as np
import argparse

def_mode = "COCO"
def_device="cpu"
def_directory = "Input/"
def_image_file = "example_image1.png"

parser = argparse.ArgumentParser(description='Run Human Pose Estimation')
parser.add_argument("--mode", default=def_mode, help="Keypoint/Skeleton Mode")
parser.add_argument("--device", default=def_device, help="Device to inference on")
parser.add_argument("--image_file", default=def_directory+def_image_file, help="Input image")
args = parser.parse_args()

print("MODE:", args.mode)
print("FILE NAME:", args.image_file)

# .prototxt file specifies the architecture of the neural network - how different layers are arranged
# .caffemodel stores the weigths of the trained model
if args.mode == "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],
                  [12,13],[0,14],[0,15],[14,16],[15,17]]

elif args.mode == "MPI":
    # protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"  # this one causes error
    protoFile = "pose/mpi/pose_deploy_linevec.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,14],[14,8],[8,9],
                  [9,10],[14,11],[11,12],[12,13]]

frame = cv2.imread(args.image_file)
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
print("IMAGE RESOLUTION: %d X %d" % (frameWidth, frameHeight))
threshold = 0.1

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU Device...")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU Device...")

t = time.time()
# inWidth and inHeight are the input image dimensions for the network
inWidth = 368
inHeight = 368
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)
output = net.forward()
print("TIME TAKEN BY NETWORK: {:.3f} seconds".format(time.time() - t))
print("OUTPUT SHAPE:", output.shape)

H = output.shape[2]
W = output.shape[3]

# Create an empty list to store the detected keypoints
points = []

for i in range(nPoints):
    # Get confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    # Draw Keypoints
    if prob > threshold: 
        cv2.circle(frameCopy, (int(x), int(y)), 7, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frameCopy, (int(x), int(y)), 4, (160, 0, 160), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(x-7), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 220), 2, lineType=cv2.LINE_AA)
        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else:
        points.append(None)
# print("NUMBER OF KEYPOINTS DETECTED:", len(points))

# Draw Skeleton
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (255, 255, 0), 3)
        cv2.circle(frame, points[partA], 8, (200, 0, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, points[partB], 8, (200, 0, 0), thickness=-1, lineType=cv2.FILLED)

cv2.imshow('Output (Keypoints)  Mode: {}'.format(args.mode), frameCopy)
cv2.imshow('Output (Skeleton)  Mode: {}'.format(args.mode), frame)
cv2.imwrite('OutputImage'+args.mode+'/'+str(args.image_file).split("/")[-1].split(".")[0]+
            '-Keypoints-'+args.mode+'.jpg', frameCopy)
cv2.imwrite('OutputImage'+args.mode+'/'+str(args.image_file).split("/")[-1].split(".")[0]+
            '-Skeleton-'+args.mode+'.jpg', frame)

print("TOTAL TIME TAKEN: {:.3f} seconds".format(time.time() - t))
cv2.waitKey(0)
