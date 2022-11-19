import numpy as np
import cv2 as cv

def readRGB(path, filename, width, height):
    f = open(path+filename, "r")
    arr = np.fromfile(f, np.uint8).reshape(height, width, 3)
    arr[:, :, [2, 0]] = arr[:, :, [0, 2]]
    return arr

def subsample(arr, size = 16):
    row, col = arr.shape
    n_row = (row-1)//size+1
    n_col = (col-1)//size+1
    n = np.zeros((n_row, n_col))
    for i in range(n_row):
        for j in range(n_col):
            n[i][j] = arr[i*size:(i+1)*size:, j*size:(j+1)*size:].mean()
    return n

imgs = []
foldername = "SAL_490_270_437"
nums = foldername.split("_")
width = int(nums[1])
height = int(nums[2])
frames = int(nums[3])
path = foldername+"/"
for i in range(frames):
    filename = foldername + "." + "{:03d}".format(i+1) + ".rgb"
    imgs.append(readRGB(path, filename, width, height))

frame1 = imgs[0]
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
for i in range(1, frames):
    next = cv.cvtColor(imgs[i], cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 16, 3, 5, 0.5, 1)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # ang = ang.reshape(-1, 10, ang.shape[1]).mean(axis = 1)
    size = 16
    subsampled_ang=subsample(ang, size)
    subsampled_mag=subsample(mag, size)
    u, c = np.unique(subsampled_ang.round(), return_counts=True)
    frequent_ang = u[c.argmax()]
    u, c = np.unique(subsampled_mag.round(1), return_counts=True)
    frequent_mag = u[c.argmax()]
    diff_area = np.argwhere(abs(subsampled_mag-frequent_mag)>0.85)
    for x,y in diff_area:
        for a in range(size):
            if x*size+a == len(imgs[i]):
                break
            for b in range(size):
                if y*size+b == len(imgs[i][0]):
                    break
                imgs[i][x*size+a][y*size+b] = [255,255,255]
    diff_area = np.argwhere(abs(subsampled_ang-frequent_ang)>0.7)
    # diff_area = np.concatenate((np.argwhere(abs(ang-frequent_ang)>0.7), np.argwhere(abs(mag-frequent_mag)>0.7)))
    for x,y in diff_area:
        for a in range(size):
            if x*size+a == len(imgs[i]):
                break
            for b in range(size):
                if y*size+b == len(imgs[i][0]):
                    break
                imgs[i][x*size+a][y*size+b] = [0,0,0]
    
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame', imgs[i])
    cv.setWindowProperty('frame', cv.WND_PROP_TOPMOST, 1)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', imgs[i])
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next
cv.destroyAllWindows()