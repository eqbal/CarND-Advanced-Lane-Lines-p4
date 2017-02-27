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

def find_initial_peaks(final_mask, bottom_pct=0.5):
    # bottom_pct: How much of the bottom to use for initial tracer placement

    shape = final_mask.shape

    bottom_sect = final_mask[-int(bottom_pct*shape[0]):, :]

    left_peak = bottom_sect[:, :int(0.5*shape[1])].sum(axis=0).argmax()
    right_peak = bottom_sect[:, int(0.5*shape[1]):].sum(axis=0).argmax() + 0.5*shape[1]

    # Return x-position of the two peaks
    return left_peak, right_peak

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

def plot_poly_orig(fitl, fitr, orig):
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
    overlay = transform.unwarp(overlay)

    # Add the overlay to the original unwarped image
    result = cv2.addWeighted(orig, 1, overlay, 0.3, 0)
    return result


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
