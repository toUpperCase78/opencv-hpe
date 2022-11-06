import cv2
import time
import numpy as np
import argparse

def_mode = "COCO"
def_device = "cpu"
def_directory = "Input/"
def_video_file = "example_video1.mp4"
frame_cnt = 1
start = time.time()

parser = argparse.ArgumentParser(description='Run Human Pose Estimation')
parser.add_argument("--mode", default=def_mode, help="Keypoint/Skeleton Mode")
parser.add_argument("--device", default=def_device, help="Device to inference on")
parser.add_argument("--video_file", default=def_directory+def_video_file, help="Input video")
args = parser.parse_args()

print("MODE:", args.mode)
print("VIDEO FILE:", args.video_file)

# .prototxt file specifies the architecture of the neural network - how different layers are arranged
# .caffemodel stores the weigths of the trained model
if args.mode == "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],
                  [12,13],[0,14],[0,15],[14,16],[15,17]]

elif args.mode == "MPI" :
    # protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"   # this one causes error
    protoFile = "pose/mpi/pose_deploy_linevec.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,14],[14,8],[8,9],
                  [9,10],[14,11],[11,12],[12,13]]

# inWidth and inHeight are the input image dimensions of every frame, for the network
inWidth = 368
inHeight = 368
threshold = 0.1

input_source = args.video_file
cap = cv2.VideoCapture(input_source)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FPS = cap.get(cv2.CAP_PROP_FPS)
FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("VIDEO RESOLUTION: {} X {}".format(W, H))
print("VIDEO FPS: {:.3f}".format(FPS))
print("TOTAL FRAMES:", FRAMES)
# hasFrame, frame = cap.read()

output_sk = 'OutputVideo'+args.mode+'/'+str(input_source).split('/')[-1].split('.')[0]+'_skeleton_'+args.mode+'.mp4'
print("VIDEO OUTPUT WILL BE WRITTEN TO", output_sk)
vid_writer = cv2.VideoWriter(output_sk, cv2.VideoWriter_fourcc('m','p','4','v'), FPS, (W, H))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device...")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device...")

while cv2.waitKey(1) < 0:
    st = time.time()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        print("TOTAL TIME TAKEN: {:.3f} seconds".format(time.time()-start))
        # print("TOTAL FRAMES:", FRAMES)
        # print(">> There are no more frames, press any key to exit now! <<")
        # cv2.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

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

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (255, 255, 0), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 7, (200, 0, 0), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 7, (200, 0, 0), thickness=-1, lineType=cv2.FILLED)

    cv2.putText(frame, "Frame: {}/{} ({:.2f} sec)".format(frame_cnt, FRAMES, time.time() - st),
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenCV-HPE on Video", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #             (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output (Keypoints)', frameCopy)
    cv2.imshow('Output (Skeleton)  Mode: {}'.format(args.mode), frame)
    vid_writer.write(frame)
    frame_cnt += 1
    # if frame_cnt % 10 == 0:
    #     print("FRAME", frame_cnt)

vid_writer.release()
