import numpy as np
import cv2 as cv


def readRGB(path, filename, width, height):
    f = open(path + filename, "r")
    arr = np.fromfile(f, np.uint8).reshape(height, width, 3)
    arr[:, :, [2, 0]] = arr[:, :, [0, 2]]
    return arr


def subsample(arr, size=16):
    row, col = arr.shape
    n_row = (row - 1) // size + 1
    n_col = (col - 1) // size + 1
    n = np.zeros((n_row, n_col))
    for i in range(n_row):
        for j in range(n_col):
            n[i][j] = arr[i * size:(i + 1) * size:, j * size:(j + 1) * size:].mean()
    return n


imgs = []
foldername = "SAL_490_270_437"
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
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

foreground_elements = []
# TODO - use neighboring frames for comparison of shape, continous shape numbering
# TODO - better feathering of shape?
for i in range(1, frames):
    next = cv.cvtColor(imgs[i], cv.COLOR_BGR2GRAY)
    motion = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 16, 3, 5, 0.5, 1)
    mag, ang = cv.cartToPolar(motion[..., 0], motion[..., 1])
    # ang = ang.reshape(-1, 10, ang.shape[1]).mean(axis = 1)
    size = 16
    subsampled_ang = subsample(ang, size)
    subsampled_mag = subsample(mag, size)
    u, c = np.unique(subsampled_ang.round(), return_counts=True)
    most_common_ang = u[c.argmax()]
    u, c = np.unique(subsampled_mag.round(1), return_counts=True)
    most_common_mag = u[c.argmax()]
    diff_area = np.concatenate((np.argwhere(abs(subsampled_ang - most_common_ang) > 0.7),
                                np.argwhere(abs(subsampled_mag - most_common_mag) > 0.85)))
    # diff_area = np.argwhere(abs(subsampled_mag-most_common_mag)>0.85)
    # for x,y in diff_area:
    #     for a in range(size):
    #         if x*size+a == len(imgs[i]):
    #             break
    #         for b in range(size):
    #             if y*size+b == len(imgs[i][0]):
    #                 break
    #             imgs[i][x*size+a][y*size+b] = [255,255,255]
    # diff_area = np.argwhere(abs(subsampled_ang-most_common_ang)>0.7)
    # for x,y in diff_area:
    #     for a in range(size):
    #         if x*size+a == len(imgs[i]):
    #             break
    #         for b in range(size):
    #             if y*size+b == len(imgs[i][0]):
    #                 break
    #             imgs[i][x*size+a][y*size+b] = [0,0,0]

    # get contiguous shapes from diff_area
    matrix = np.zeros_like(subsampled_ang)
    total_shapes = 0
    for x, y in diff_area:
        matrix[x][y] = 1
    m = len(matrix)
    n = len(matrix[0])

    shapes = {}

    for x, y in diff_area:
        stack = [(x, y)]
        area = 0
        coord = set()
        visited = set()
        while stack:
            u, v = stack.pop()
            if 0 <= u < m and 0 <= v < n and (u, v) not in visited:
                visited.add((u, v))
                if matrix[u][v] == 1:
                    area += 1
                    coord.add((u, v))
                    stack.append((u - 1, v))
                    stack.append((u + 1, v))
                    stack.append((u, v - 1))
                    stack.append((u, v + 1))
        if area > 0:
            if len(coord) > 0:
                shapes[total_shapes] = coord
                total_shapes += 1

    # render only top objects
    # if shapes.values():
    #     largest_shape = max(shapes.values(), key=len)
    #     for x,y in largest_shape    :
    #         for a in range(size):
    #             if x*size+a == len(imgs[i]):
    #                 break
    #             for b in range(size):
    #                 if y*size+b == len(imgs[i][0]):
    #                     break
    #                 imgs[i][x*size+a][y*size+b] = [0,0,0]
    # foreground_frame = np.full((len(imgs[i]), len(imgs[i][0]), 3), 255)
    foreground_frame = np.full_like(imgs[i], 255)
    for shape in shapes.values():
        if len(shape) > 8:
            for x, y in shape:
                for a in range(size):
                    if x * size + a == len(imgs[i]):
                        break
                    for b in range(size):
                        if y * size + b == len(imgs[i][0]):
                            break
                        foreground_frame[x * size + a][y * size + b] = imgs[i][x * size + a][y * size + b]

    # for x in range(len(mask)):
    #     for y in range(len(mask[0])):

    # TODO - idea
    # compare each region with the frame before(maybe 2 to be safe)
    # same/similar regions will be inherit number throughout the video
    # region may terminate, or new ones will begin
    # in the end will receive each object number and frame data.

    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame', foreground_frame)
    cv.setWindowProperty('frame', cv.WND_PROP_TOPMOST, 1)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', imgs[i])
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next
cv.destroyAllWindows()