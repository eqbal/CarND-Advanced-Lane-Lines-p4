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


#### 4. Perspective Transform

For this section I wrote `PerspectiveTransformer` class, which allow me to wrap/unwrap perspective in single line while only having to compute the transformation matrix once.

I used `cv2.getPrespectiveTranform` to compute the transformation and inverse it. And then applied the images using `cv2.warpPerspective`. The code looks like this:

```python
class PerspectiveTransformer():
    def __init__(self, src, dist):
        self.Mpersp = cv2.getPerspectiveTransform(src, dst)
        self.Minv   = cv2.getPerspectiveTransform(dst, src)

    # Apply perspective transform
    def warp(self, img):
        return cv2.warpPerspective(img, self.Mpersp, (img.shape[1], img.shape[0]))

    # Reverse perspective transform
    def unwarp(self, img):
        return cv2.warpPerspective(img, self.Minv, (img.shape[1], img.shape[0]))
```

For my transformation, I went with with the following `src` and `dst` points:

```python
src = np.array([[585, 460], [203, 720], [1127, 720], [695, 460]]).astype(np.float32)
dst = np.array([[320, 0], [320, 720], [960, 720], [960, 0]]).astype(np.float32)
```


![Dist1](./assets/transformation.png)


#### 5. Detect Lane Lines

This stage is where we will try to extract the actual lane pixels for both the left and right lanes from the images.

First, in order to know where the lane line begin, I created a histogram of the masked pixels in the bottom half of the image, and selected the biggest peak in both the left and right sides.

![peak-lane](./assets/find_lane_1.png)

The second step was to use `numpy` to split the image into a specific number of chunks (10 to be exact). In each chuck, we do:

  - Take the mean pixel x-position of the last chuck 
  - Select all masked pixels within 80 pixels of this value
  - Add coordinates of these pixels on an array 

So, I ended up with an array of (x,y) values for both the left and the right lane line. 

I used `numpy.polyfit` to then fit a quadratic curve to each lane line, which can then be plotted on the input frame.

To smooth out the final curve result, I took a weighted mean with the last frame's polynomial coefficients (`w=0.2`).

![peak-lane-2](./assets/lane_line_2.png)

For more details check out the `helpers.py` file:

```python
# helpers.py

# Find the peaks of the bottom half, for sliding window analysis
def find_initial_peaks(final_mask, bottom_pct=0.5):
    # bottom_pct: How much of the bottom to use for initial tracer placement
    
    shape = final_mask.shape
    
    bottom_sect = final_mask[-int(bottom_pct*shape[0]):, :]
    
    left_peak = bottom_sect[:, :int(0.5*shape[1])].sum(axis=0).argmax()
    right_peak = bottom_sect[:, int(0.5*shape[1]):].sum(axis=0).argmax() + 0.5*shape[1]
    
    # Return x-position of the two peaks
    return left_peak, right_peak

# This applies the sliding window approach to find lane pixels, and then fits a polynomial to the found pixels.
def sliding_window_poly(final_mask, left_peak, right_peak, num_chunks=10, leeway=80):
    # num_chunks: Number of chunks to split sliding window into
    # leeway: Number of pixels on each side horizontally to consider
    
    # Split the image vertically into chunks, for analysis.
    chunks = []
    assert final_mask.shape[0] % num_chunks == 0, 'Number of chunks must be a factor of vertical resolution!'
    px = final_mask.shape[0] / num_chunks # Pixels per chunk
    for i in range(num_chunks):
        chunk = final_mask[i*px:(i+1)*px, :]
        chunks.append(chunk)

    # Reverse the order of the chunks, in order to work from the bottom up
    chunks = chunks[::-1]
    
    # Loop over chunks, finding the lane centre within the leeway.
    lefts = [left_peak]
    rights = [right_peak]
    
    left_px, left_py, right_px, right_py = [], [], [], []
    
    for i, chunk in enumerate(chunks):
        offset = (num_chunks-i-1)*px
        
        last_left = int(lefts[-1])
        last_right = int(rights[-1])
        
        # Only consider pixels within +-leeway of last chunk location
        temp_left_chunk = chunk.copy()
        temp_left_chunk[:, :last_left-leeway] = 0
        temp_left_chunk[:, last_left+leeway:] = 0
        
        temp_right_chunk = chunk.copy()
        temp_right_chunk[:, :last_right-leeway] = 0
        temp_right_chunk[:, last_right+leeway:] = 0
        
        # Save the x, y pixel indexes for calculating the polynomial
        left_px.append(temp_left_chunk.nonzero()[1])
        left_py.append(temp_left_chunk.nonzero()[0] + offset)
        
        right_px.append(temp_right_chunk.nonzero()[1])
        right_py.append(temp_right_chunk.nonzero()[0] + offset)
    
    # Create x and y indice arrays for both lines
    left_px = np.concatenate(left_px)
    left_py = np.concatenate(left_py)
    right_px = np.concatenate(right_px)
    right_py = np.concatenate(right_py)
    
    # Fit the polynomials!
    l_poly = np.polyfit(left_py, left_px, 2)
    r_poly = np.polyfit(right_py, right_px, 2)
    
    return l_poly, r_poly

```

#### 6. Determine Lane Curvature

To calculate the lane curvature radius, I scaled the x and y of my lane pixel and then fit a new polynomial to the data. Using the new polynomial, I could the use the radius of curvature formula to calculate the curve radius in metres at the base of the image.

I performed this on both the left and right lane line and then took the mean of these values for my final curvature value. I took a weighted average with calculated curvature of the last frame, allowing the displayed curvature value to be more accurate and smooth representation.

Check out the code I used below: 

```python
# helpers.py

def get_curvature(poly, mask):
    yscale = 30 / 720 # Real world metres per y pixel
    xscale = 3.7 / 700 # Real world metres per x pixel

    # Convert polynomial to set of points for refitting
    ploty = np.linspace(0, mask.shape[0]-1, mask.shape[0])
    fitx = poly[0] * ploty ** 2 + poly[1] * ploty + poly[2]

    # Fit new polynomial
    fit_cr = np.polyfit(ploty * yscale, fitx * xscale, 2)

    # Calculate curve radius
    curverad = ((1 + (2 * fit_cr[0] * np.max(ploty) * yscale + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    return curverad
```

#### 7. Lane Position

To compute the Position of the car in the lane, I used the lane polyfit to find the bottom position of the left and right lanes respectively. Assuming the width of the lane was 3.7 metres, I calculated the scale of the transformed image, and then used the distance between the centre of the image and the centre of the lane to calculate the offset of the car. 

I did not use weighted mean with the last frame as I found the offset value to be more stable than the lane curvature.

Check out the code related: 

```python
# helpers.py

def find_offset(l_poly, r_poly):
    lane_width = 3.7  # metres
    h = 720  # height of image (index of image bottom)
    w = 1280 # width of image

    # Find the bottom pixel of the lane lines
    l_px = l_poly[0] * h ** 2 + l_poly[1] * h + l_poly[2]
    r_px = r_poly[0] * h ** 2 + r_poly[1] * h + r_poly[2]

    # Find the number of pixels per real metre
    scale = lane_width / np.abs(l_px - r_px)

    # Find the midpoint
    midpoint = np.mean([l_px, r_px])

    # Find the offset from the centre of the frame, and then multiply by scale
    offset = (w/2 - midpoint) * scale
    return offset

```

### The final Pipeline

To hide the implementation of different methods I ended up creating 'LaneLineFinder' class which has `process_frame` main method that receives an image and apply the steps described above. The final implementation of this class looks like this: 

```python
#lane_line_finder.py


import cv2
from camera_calibrate import *
from helpers import *
from prospective_transformer import *

last_rad = None
last_l_poly = None
last_r_poly = None

class LaneLineFinder():

    def __init__(self):

        self.cc  = self.set_cam_calibration()
        self.rad_alpha = 0.05
        self.poly_alpha = 0.2

        src = np.array([
                [585, 460],
                [203, 720],
                [1127, 720],
                [695, 460]]
              ).astype(np.float32)

        dst = np.array([
                [320, 0],
                [320, 720],
                [960, 720],
                [960, 0]]
              ).astype(np.float32)



        self.transform = PerspectiveTransformer(src, dst)

    def process_frame(self, img):

        global last_rad, last_l_poly, last_r_poly

        # Undistort the image using the camera calibration
        img = self.cc.undistort(img)

        # Keep the untransformed image for later
        orig = img.copy()

        # Apply perspective transform to the image
        img = self.transform.warp(img)

        # Apply the HLS/Sobel mask to detect lane pixels
        mask = mask_image(img)

        # Find initial histogram peaks
        left_peak, right_peak = find_initial_peaks(mask)

        # Get the sliding window polynomials for each line line
        l_poly, r_poly = sliding_window_poly(mask, left_peak, right_peak, leeway=80)

        # Update polynomials using weighted average with last frame
        if last_l_poly is None:
            # If first frame, initialise buffer
            last_l_poly = l_poly
            last_r_poly = r_poly
        else:
            # Otherwise, update buffer
            l_poly = (1 - self.poly_alpha) * last_l_poly + self.poly_alpha * l_poly
            r_poly = (1 - self.poly_alpha) * last_r_poly + self.poly_alpha * r_poly
            last_l_poly = l_poly
            last_r_poly = r_poly

        # Calculate the lane curvature radius
        l_rad = get_curvature(l_poly, mask)
        r_rad = get_curvature(r_poly, mask)

        # Get mean of curvatures
        rad = np.mean([l_rad, r_rad])

        # Update curvature using weighted average with last frame
        if last_rad is None:
            last_rad = rad
        else:
            last_rad = (1 - self.rad_alpha) * last_rad + self.rad_alpha * rad

        # Create image
        final = self.plot_poly_orig(l_poly, r_poly, orig)

        # Write radius on image
        cv2.putText(final, 'Lane Radius: {}m'.format(int(last_rad)),
                (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, 255)

        # Write lane offset on image
        offset = find_offset(l_poly, r_poly)
        cv2.putText(final, 'Lane Offset: {}m'.format(round(offset, 4)),
                (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, 255)

        return final

    def set_cam_calibration(self):
        cc = CameraCalibrate(hcount=6, vcount=9)
        cc.get_images()
        cc.call()
        return cc


    def plot_poly_orig(self, fitl, fitr, orig):
        # Draw lines from polynomials
        ploty = np.linspace(0, orig.shape[0]-1, orig.shape[0])
        fitl = fitl[0]*ploty**2 + fitl[1]*ploty + fitl[2]
        fitr = fitr[0]*ploty**2 + fitr[1]*ploty + fitr[2]

        pts_left = np.array([np.transpose(np.vstack([fitl, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([fitr, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Create an overlay from the lane lines
        overlay = np.zeros_like(orig).astype(np.uint8)
        cv2.fillPoly(overlay, np.int_([pts]), (0,255, 0))

        # Apply inverse transform to the overlay to plot it on the original road
        overlay = self.transform.unwarp(overlay)

        # Add the overlay to the original unwarped image
        result = cv2.addWeighted(orig, 1, overlay, 0.3, 0)
        return result

```

To generate the video (check out `playground` notebook):


```python

from lane_line_finder import *
lane_fider = LaneLineFinder()

output_video = 'project_video_out.mp4'
input_video  = VideoFileClip("project_video.mp4")

white_clip = input_video.fl_image(lane_fider.process_frame)

%time white_clip.write_videofile(output_video, audio=False)
```


The `process_frame` function, outputs the original image with the lane area drawn on top, and the curvature radius and lane offset in the top left corner. 

![pipeline](./assets/pipeline.png)

#### Video Output

The final result of my algorithm for this project video can be found here: 

[![Advanced Lane Line Detection](http://img.youtube.com/vi/xmYNKJdVdz4/0.jpg)](http://www.youtube.com/watch?v=xmYNKJdVdz4)

### The final flow:

Here is the flow that I'm using for lane detection: 

![flow](./assets/flow.png)

#### Conclusion


This was an absolutely an awesome project that needed alot of engineering. At first the lanes were all over the place and very jittery. I applied techniques to help such as make sure that the polygon that comes after was of similar shape, I also ruled out polygons that failed to meet area requirements. When there errors across the last correct polygon was written instead.

Future improvements to avoid the algorithm from failing is a much cleaner bit mask extraction. When tested on the most difficult video the main problem was additional noise creating points that ruined the polynomial fit. With more time a better bit mask extraction can be applied to enhance this algorithm.


