# opencv-hpe

**Human Pose Estimation with OpenCV using Python**

## Overview

Perform "human pose estimation" on images and videos with OpenCV using Python language. There are two separate files (**get_hpe_image.py** & **get_hpe_video.py**) to fulfill these tasks.

_Please note that it only works for single person (for multiple ones, weird results may appear!)..._

**Example inputs and outputs can be inspected in their corresponding directories beforehand.**

Everything provided here was tested for **Python 3.10**, **numpy 1.23** and **opencv-python 4.6**. Thus, I recommend to install the lastest version of Python and the related components in order to run the scripts smoothly.

## Source

This work is heavily based on this article: [Deep Learning base Human Pose Estimation using OpenCV](https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/) 

And here, the scripts were slightly changed to make them more informative to the user while being executed and fix some issues (especially for the videos).

## Important!

Before you start, keep in mind that the files for model weights (.caffemodel) are not available in this repo! You have to run the script below in order to get these files downloaded in their respective directories (each of them has size of 200+ MB):

```
./getModels.sh
```

## How to Run

The scripts for image and video can be run without entering any arguments, so they'll use the default values defined in the codebase inside. e.g.:

```
python get_hpe_image.py

python get_hpe_video.py
```

However, it is highly advised to use the supported additional arguments for getting different results. Here are the usage examples:

```
python get_hpe_image.py --mode <enter_mode> --device <enter_device> --image_file <input_image_location>

python get_hpe_video.py --mode <enter_mode> --device <enter_device> --video_file <input_video_location>
```

`--mode` : Set the keypoint/skeleton mode ("COCO" or "MPI") (default: "COCO")

Remember, COCO has 18 keypoints to build the skeleton on the estimated human. These are: _0: Nose, 1: Neck, 2: Right Shoulder, 3: Right Elbow, 4: Right Wrist, 5: Left Shoulder, 6: Left Elbow, 7: Left Wrist, 8: Right Hip, 9: Right Knee, 10: Right Ankle, 11: Left Hip, 12: Left Knee, 13: Left Ankle, 14: Right Eye, 15: Left Eye, 16: Right Ear, 17: Left Ear_

Likewise, MPI mode contains 15 keypoints for pose estimation. These are: _0: Head, 1: Neck, 2: Right Shoulder, 3: Right Elbow, 4: Right Wrist, 5: Left Shoulder, 6: Left Elbow, 7: Left Wrist, 8: Right Hip, 9: Right Knee, 10: Right Ankle, 11: Left Hip, 12: Left Knee, 13: Left Ankle, 14: Chest_

`--device` : Set the device to inference on ("cpu" or "gpu") (default: "cpu")

`--image_file` : Define the location for the input image, typing the directory and the file name together is highly recommended (default: "Input/example_image1.png")

`--video_file` : Define the location for the input video, typing the directory and the file name together is highly recommended (default: "Input/example_video1.mp4")

When pose estimation is completed, the output will be stored to the directory corresponding to the selection of mode and input type.

## Performance

For any image or every frame for any video, it takes **1.3-1.5 seconds** to complete the pose estimation with Intel Core i7-1065G7 (4C/8T) processor. Therefore, less times can be seen with more powerful CPU in terms of the gen. of architecture, clock speed, # of cores.

Note that the time taken may also change depending on the input dimensions for the network and/or the input resolution...

**Caution!** When run on CPU, the processor runs at 100% load almost all the time, hence you might encounter high temperature!

Moreover, I haven't tested on GPU mode yet; but when it is carried out in the future, the details and effects will be explained here...
