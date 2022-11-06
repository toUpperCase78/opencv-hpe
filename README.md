# opencv-hpe

**Human Pose Estimation with OpenCV using Python**

## Overview

Perform "human pose estimation" on images and videos with OpenCV using Python language. There are two separate files (**get_hpe_image.py** & **get_hpe_video.py**) to fulfill these tasks.

_Please note that it only works for single person (for multiple ones, weird results may appear!)..._

**Example inputs and outputs can be inspected in their corresponding directories beforehand.**

Everything provided here was tested for **Python 3.10**, **numpy 1.23** and **opencv-python 4.6**. Thus, I recommend to install the lastest version of Python and the related components in order to run the scripts smoothly.

## Source

This work is heavily based on this article: [Deep Learning base Human Pose Estimation using OpenCV](https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/) 

And here, the scripts were slightly changed to make them more informative while being used and fix some issues (especially for the videos).

## How to Run

The scripts for image and video can be run without entering any arguments, so they'll use the default values defined inside. e.g.:

```
python get_hpe_image.py

python get_hpe_video.py
```

## Performance

For any image or every frame for any video, it takes **1.3-1.5 seconds** to complete the pose estimation with Intel Core i7-1065G7 (4C/8T) processor. Therefore, less times can be seen with more powerful CPU in terms of the gen. of architecture, clock speed, # of cores.

Note that the time taken may also change depending on the input dimensions for the network and/or the input resolution...

**Caution!** When run on CPU, the processor runs at 100% load almost all the time, hence you might encounter high temperature!

Moreover, I haven't tested on GPU yet, but when it is carried out in the future, the details will be explained here...
