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
    row, col, depth = arr.shape
    n_row = (row - 1) // size + 1
    n_col = (col - 1) // size + 1
    n = np.zeros((n_row, n_col,depth))
    for i in range(n_row):
        for j in range(n_col):
            for k in range(depth):
                n[i][j][k] = arr[i * size:(i + 1) * size:, j * size:(j + 1) * size:,k].mean()
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

motion_list = []
mag_list = []
ang_list = []
subsampled_motion_list = []
subsampled_mag_list = []
subsampled_ang_list = []


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
    subsampled_mag, subsampled_ang = cv.cartToPolar(subsampled_motion[..., 0], subsampled_motion[..., 1])

    # store all data in lists
    motion_list.append(motion)
    mag_list.append(mag)
    ang_list.append(ang)
    subsampled_motion_list.append(subsampled_motion)
    subsampled_mag_list.append(subsampled_mag)
    subsampled_ang_list.append(subsampled_ang)


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