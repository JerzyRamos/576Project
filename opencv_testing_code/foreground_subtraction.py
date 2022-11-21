import numpy as np
import cv2 as cv


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

foreground_frames = []
background_frames = []
# TODO - use neighboring frames for comparison of shape, continuous shape numbering
# thoughts
# use variable f to control number of frames on either side, like f=2 -> 2+f+2
# if black block all exists, then average all five, every block above quantile_threshold will exist
# TODO - better feathering of shape?
for i in range(1, frames):
    next = cv.cvtColor(imgs[i], cv.COLOR_BGR2GRAY)
    motion = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 16, 3, 5, 0.5, 1)
    # mag, ang = cv.cartToPolar(motion[..., 0], motion[..., 1], angleInDegrees=True)
    size = 16
    subsampled_motion = subsample(motion, size)
    subsampled_mag, subsampled_ang = cv.cartToPolar(subsampled_motion[..., 0], subsampled_motion[..., 1], angleInDegrees=True)
    subsampled_ang = subsampled_ang / 72
    m, n = subsampled_ang.shape
    u, c = np.unique(subsampled_ang.round(), return_counts=True)
    most_common_ang = u[c.argmax()]
    u, c = np.unique(subsampled_mag.round(1), return_counts=True)
    most_common_mag = u[c.argmax()]
    diff_area = []
    for x in range(m):
        for y in range(n):
            if min(abs(subsampled_ang[x][y] - most_common_ang), abs(5 - subsampled_ang[x][y] - most_common_ang)) > 0.5:
                diff_area.append((x,y))


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
                    # stack.append((u + 1, v + 1))
                    # stack.append((u + 1, v - 1))
                    # stack.append((u - 1, v + 1))
                    # stack.append((u - 1, v - 1))

        if area > 0:
            if len(coord) > 2:
                shapes[total_shapes] = visited
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
    background_frame = imgs[i].copy()
    for shape in shapes.values():
        if len(shape) > 0:
            for x, y in shape:
                for a in range(size):
                    if x * size + a == len(imgs[i]):
                        break
                    for b in range(size):
                        if y * size + b == len(imgs[i][0]):
                            break
                        foreground_frame[x * size + a][y * size + b] = imgs[i][x * size + a][y * size + b]
                        background_frame[x * size + a][y * size + b] = [0, 0, 0]


    # for rendering diff_area directly
    # foreground_frame = np.full_like(imgs[i], 255)
    # for x, y in diff_area:
    #     for a in range(size):
    #         if x * size + a == len(imgs[i]):
    #             break
    #         for b in range(size):
    #             if y * size + b == len(imgs[i][0]):
    #                 break
    #             foreground_frame[x * size + a][y * size + b] = imgs[i][x * size + a][y * size + b]

    # for x in range(len(mask)):
    #     for y in range(len(mask[0])):

    # TODO - idea
    # compare each region with the frame before(maybe 2 to be safe)
    # same/similar regions will be inherit number throughout the video
    # region may terminate, or new ones will begin
    # in the end will receive each object number and frame data.

    # for rendering subsampled_ang
    # for x in range(m):
    #     for y in range(n):
    #         for a in range(size):
    #             if x * size + a == len(imgs[i]):
    #                 break
    #             for b in range(size):
    #                 if y * size + b == len(imgs[i][0]):
    #                     break
    #                 hsv[x * size + a][y * size + b][0] = subsampled_ang[x][y]*36
    foreground_frames.append(foreground_frame)
    background_frames.append(background_frame)

    # hsv[..., 0] = ang.round() * 36
    hsv[..., 2] = 155
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame', foreground_frame)
    cv.setWindowProperty('frame', cv.WND_PROP_TOPMOST, 1)
    k = cv.waitKey(30) & 0xff
    # print(i)
    # k = cv.waitKey(0)
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', imgs[i])
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next
cv.destroyAllWindows()