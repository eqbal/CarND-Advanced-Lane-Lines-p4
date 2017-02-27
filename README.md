# Advance Lane Finding Documentation
#### By: [Eqbal Eki](http://www.eqbalq.com/)

****

###Steps:
    
1. Camera Calibration
2. Distortion Correction
3. Binary Thresholding
4. Perspective Transform
5. Detect Lane Lines
6. Determine Lane Curvature
7. Impose Lane Boundaries on Original Image
8. Output Visual Display of Lane Boundaries and Numerical Estimation of Lane Curvature and Vehicle Position

###Flow:

Here is the flow that I'm using for lane detection: 

... to be filled later on when the final module built

#### 1. Camera Calibration

In this stage, I read in a series of test images from the camera_cal directory. Each image is a view of a chessboard pattern at various angles.

I use the OpenCV function findChessBoardCorners() to find the x,y coordinates of each corner on a given image. Once I have an array with all these corners, I use the OpenCV calibrateCamera() function to calculate the camera matrix, the vector of distortion coefficients, and the camera vectors of rotation and translation.

To go more in details of how I implemented this. I created `CameraCalibrate` class, this way we keep the code more organized and reusable, I used [SRP](https://en.wikipedia.org/wiki/Single_responsibility_principle) principle so we usually the class has one main thing to do, in this case as the name indicate it calibrate the camera.

Another added value of using Class is that this way I'll only need to compute the matrix and coefficients once and it can then be applied easily at any point. 

It works like this:

```python
# Calibrate the camera and create camera matrix and distortion coefficients
cc = CameraCalibrate(hcount=6, vcount=9)
# Get the images from /camera_cal folder
cc.get_images()
cc.call()
# Run it in new images
cc.undistort(img)
```

When initialized we provide the size of the chessboard (in our case it's 9X6). 

To compute the camera matrix and distortion coefficients, `CameraCalibrate` uses OpenCV's `findChessBoardCorners` function on grayscale version of each input image, ti generate set of image points. 

The class will skip any images that have parts of the chessboard cut-off, printing a warning (some images in the dataset has this issue)


#### 2. Undistort Image

Once we have calibrated the camera as above, we can now apply the camera matrix and distortion coefficients to correct distortion effects on camera input images. This is done using the OpenCV 'undistort()' function.

I use `numpy.mgrid` in order to create a matrix of (x, y, z) object points. Once I have the image points and the matching object points for the entire calibration dataset, I then use `cv2.calibrateCamera` to create the camera matrix and distortion coefficients. These variables are stored inside the class, until `calibraion.undistort(img)` is called, at which point they are passed into `cv.undistort`, returning an undistorted version of the user's input image

![Dist1](./assets/dist1.png)


![Dist1](./assets/dist2.png)


#### 3. Binary Thresholding

The Thresholding stage is where we process the undistorted image and attempt to mask out which pixels are part of the lanes and remove those that are not. 

To achieve this, I used combination of different transforms.

First, I used `cv2.cvtColor()` to convert to HLS space, where I created a binary mask, detecting pixels where hue < 100 and saturation > 100 as lane line pixels.

After this, I also created a mask which used a threshold on absolute Sobel magnitude. I decided to go with mask which applied different Sobel thresholds to the top and bottom halves of the image. as the top half of the image had smoother gradients due to the prospective transform. At the end, I selected any pixel > 10 sobel magnitude at the top of the image, and > 35 at the bottom of the image.

I found that the HLS threshold worked better closer to the car, and the Sobel threshold worked better further away. So, a bitwise `OR` of these two thresholds gave me one that I was very happy with.

The binary thresholding code can be found in the `mask_image` method in `helpers.py` file.


![Dist1](./assets/thresholding.png)

