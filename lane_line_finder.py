from camera_calibrate import *
from helpers import *
from prospective_transformer import *

class LaneLineFinder():

    def __init__(self):
        self.cc  = self.set_cam_calibration()

        self.src = np.array([
            [585, 460],
            [203, 720],
            [1127, 720],
            [695, 460]]
        ).astype(np.float32)

        self.dst = np.array([
            [320, 0],
            [320, 720],
            [960, 720],
            [960, 0]]
        ).astype(np.float32)

        self.transform = self.set_perspective_transformer()

        self.last_rad = None
        self.last_l_poly = None
        self.last_r_poly = None
        self.rad_alpha = 0.05
        self.poly_alpha = 0.2


    def process_frame(self, img):

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
        final = plot_poly_orig(l_poly, r_poly, orig)

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
        return cc.call()

    def set_perspective_transformer(self):
        return PerspectiveTransformer(self.src, self.dst)


