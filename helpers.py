def mask_image(img):
    img = img.copy()

    # Apply a mask on HLS colour channels
    # This selects pixels with higher than 100 saturation and lower than 100 hue
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    mask = np.zeros_like(hls[:, :, 0])
    mask[(hls[:, :, 2] > 100) & (hls[:, :, 0] < 100)] = 1

    # Apply a sobel magnitude threshold
    # I apply a more lenient mag_thresh to the upper part of the transformed image, as this part is blurrier
    # and will therefore have smoother gradients.
    # On the bottom half, this selects pixels with >10 sobel magnitude, and on the top half,
    # selects pixels with >35 sobel magnitude
    upper_mag = mag_thresh(img, 3, (10, 255))
    lower_mag = mag_thresh(img, 3, (35, 255))

    mag_mask = np.zeros_like(lower_mag)
    mag_mask[:int(mag_mask.shape[0]/2), :] = upper_mag[:int(mag_mask.shape[0]/2), :]
    mag_mask[int(mag_mask.shape[0]/2):, :] = lower_mag[int(mag_mask.shape[0]/2):, :]

    # Use the bitwise OR mask of both masks for the final mask
    final_mask = np.maximum(mag_mask, mask)

    # Return the transformed mask
    return final_mask


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag / scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output
