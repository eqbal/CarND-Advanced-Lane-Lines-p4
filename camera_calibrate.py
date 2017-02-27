import numpy as np
import cv2
import matplotlib.image as mpimg
import glob

class CameraCalibrate():

    def __init__(self, hcount, vcount):
        self.mtx = None
        self.dist = None
        self.size = None
        self.hcount = hcount
        self.vcount = vcount
        self.calibrated = False

    def call(self):
        objpoints, imgpoints = self.get_all_points()
        self.mtx, self.dist  = self.calibrate(objpoints, imgpoints)

    def get_images(self):
        self.imgs = [mpimg.imread(f) for f in sorted(
            glob.glob('./camera_cal/*.jpg'))]

    def gridspace(self):
        obj = np.zeros((self.hcount * self.vcount, 3)).astype(np.float32)
        obj[:, :2] = np.mgrid[0:self.hcount, 0:self.vcount].T.reshape(-1, 2)
        return obj

    def get_points(self, img):
        objp = self.gridspace()
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.size = grey.shape[::-1]
        ret, corners = cv2.findChessboardCorners(
                grey, (self.hcount, self.vcount), None)

        if ret:
            return corners, objp
        else:
            return None, None

    def get_all_points(self):
        objpoints = []
        imgpoints = []

        for i, img in enumerate(self.imgs):
            imgp, objp = self.get_points(img)
            if imgp is None:
                print('[CameraCalibration] Image {} skipped during calibrations'.format(i))
            else:
                objpoints.append(objp)
                imgpoints.append(imgp)

        return objpoints, imgpoints

    def calibrate(self, objpoints, imgpoints):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, self.size, None, None)
        return mtx, dist

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
