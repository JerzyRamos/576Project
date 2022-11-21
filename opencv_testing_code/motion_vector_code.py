import numpy as np
import cv2 as cv

# This is a part of the foreground_subtraction.py code.
# This code provides all motion vector and angle&magnitude data.

def readRGB(path, filename, width, height):
    f = open(path + filename, "r")
    arr = np.fromfile(f, np.uint8).reshape(height, width, 3)
    arr[:, :, [2, 0]] = arr[:, :, [0, 2]]
    return arr


def subsample(arr, size=16):
    row, col = arr.shape[0], arr.shape[1]
    n_row = (row - 1) // size + 1
    n_col = (col - 1) // size + 1
    if len(arr.shape) == 3:
        d = arr.shape[2]
        n = np.zeros((n_row, n_col,d))
        for i in range(n_row):
            for j in range(n_col):
                for k in range(d):
                    n[i][j][k] = np.mean([x[k] for x in arr[i * size:(i + 1) * size:, j * size:(j + 1) * size:]])
    else:
        n = np.zeros((n_row, n_col))
        for i in range(n_row):
            for j in range(n_col):
                n[i][j] = arr[i * size:(i + 1) * size:, j * size:(j + 1) * size:].mean()
    return n


# preparation for reading rgb files
foldername = "SAL_490_270_437"
imgs = []
nums = foldername.split("_")
width = int(nums[1])
height = int(nums[2])
frames = int(nums[3])
path = foldername + "/"
for i in range(frames):
    filename = foldername + "." + "{:03d}".format(i + 1) + ".rgb"
    imgs.append(readRGB(path, filename, width, height))

frame1 = imgs[0]
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

for i in range(1, frames):
    next = cv.cvtColor(imgs[i], cv.COLOR_BGR2GRAY)
    # motion is the motion vector of each pixel in the frame
    motion = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 16, 3, 5, 0.5, 1)

    # this converts the cartesian value(x,y pairs) into angular and magnitude data, which may be helpful
    mag, ang = cv.cartToPolar(motion[..., 0], motion[..., 1])

    # subsample motion vectors into fewer blocks by averaging, like professor's macro blocks
    # use 'size' to control the height/width in pixels of each motion vector block
    size = 16
    subsampled_motion = subsample(motion, size)
    subsampled_ang = subsample(ang, size)
    subsampled_mag = subsample(mag, size)


    # code for displaying the frames
    cv.imshow('frame', imgs[i])
    cv.setWindowProperty('frame', cv.WND_PROP_TOPMOST, 1)
    k = cv.waitKey(30) & 0xff
    # press esc to stop the playback
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', imgs[i])
    prvs = next
cv.destroyAllWindows()