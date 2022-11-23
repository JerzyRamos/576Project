from collections import defaultdict
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
    n = np.zeros((n_row, n_col, depth))
    for i in range(n_row):
        for j in range(n_col):
            for k in range(depth):
                n[i][j][k] = arr[i * size:(i + 1) * size:, j * size:(j + 1) * size:, k].mean()
    return n


# change args to use preset values for different videos

# args = ["SAL_490_270_437", 1.1, 5, 0, 4, 2, 2]
args = ["Stairs_490_270_346", 0.4, 15, 0, 8, 5, 0]
foldername = args[0]
ang_threshold = args[1]
nearest_frames = args[2]
quantile_threshold = args[3]
min_shape_threshold = args[4]
horizontal_edge_buffer = args[5]
vertical_edge_buffer = args[6]

print("Program started")

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
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

foreground_frames = []
background_frames = []
# frame_shapes = []
# frame_shapes_dict = []
frame_masks = []
# TODO - use neighboring frames for comparison of shape, continuous shape numbering
# thoughts
# use variable f to control number of frames on either side, like f=2 -> 2+f+2
# if black block all exists, then average all five, every block above quantile_threshold will exist
# TODO - better feathering of shape?
print("Frames processing started")
for i in range(1, len(imgs)):
    if i == len(imgs) // 4:
        print("Processing 25% done")
    elif i == len(imgs) // 2:
        print("Processing 50% done")
    elif i == len(imgs) // 4 * 3:
        print("Processing 75% done")
    nxt = cv.cvtColor(imgs[i], cv.COLOR_BGR2GRAY)
    motion = cv.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 16, 3, 5, 0.5, 1)
    # mag, ang = cv.cartToPolar(motion[..., 0], motion[..., 1], angleInDegrees=True)
    size = 16
    subsampled_motion = subsample(motion, size)
    subsampled_mag, subsampled_ang = cv.cartToPolar(subsampled_motion[..., 0], subsampled_motion[..., 1],
                                                    angleInDegrees=True)
    subsampled_ang = subsampled_ang / 72
    m, n = subsampled_ang.shape
    u, c = np.unique(subsampled_ang.round(), return_counts=True)
    most_common_ang = u[c.argmax()]
    # u, c = np.unique(subsampled_mag.round(1), return_counts=True)
    # most_common_mag = u[c.argmax()]

    # diff_area = []
    mask = np.zeros_like(subsampled_ang)

    for x in range(vertical_edge_buffer, m-vertical_edge_buffer):
        for y in range(horizontal_edge_buffer, n-horizontal_edge_buffer):
            if min(abs(subsampled_ang[x][y] - most_common_ang),
                   abs(5 - subsampled_ang[x][y] - most_common_ang)) > ang_threshold:
                # TODO - Try to include magnitude difference into segmentation
                # abs(subsampled_mag[x][y]-most_common_mag)>2:
                # diff_area.append((x,y))
                mask[x][y] = 1
    frame_masks.append(mask)
    prvs = nxt

video = cv.VideoWriter('foreground_video.mp4',cv.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

for i in range(len(imgs)-1):
    matrix = np.mean(frame_masks[max(0, i - nearest_frames):i + nearest_frames], axis=0)
    # frame_masks[i] = np.zeros(frame_masks[i].shape)
    # get contiguous shapes from diff_area
    total_shapes = 0
    shapes = {}
    m, n = frame_masks[i].shape

    for x in range(m):
        for y in range(n):
            if matrix[x][y] == 0:
                continue
            stack = [(x, y)]
            area = 0
            coord = set()
            visited = set()
            while stack:
                u, v = stack.pop()
                if 0 <= u < m and 0 <= v < n and (u, v) not in visited:
                    visited.add((u, v))
                    if matrix[u][v] > quantile_threshold:
                        # frame_masks[i][u][v] = 1
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
                    shapes[total_shapes] = coord
                    total_shapes += 1

    foreground_frame = np.full_like(imgs[i], 255)
    background_frame = imgs[i].copy()
    shapes_dict = {}
    for shape in shapes.values():
        if len(shape) > min_shape_threshold:
            for x, y in shape:
                shapes_dict[(x, y)] = shape
                foreground_frame[x * size:(x + 1) * size, y * size:(y + 1) * size] = imgs[i][x * size:(x + 1) * size,
                                                                                     y * size:(y + 1) * size]
                background_frame[x * size:(x + 1) * size, y * size:(y + 1) * size] = [0, 0, 0]

    # frame_shapes.append(shapes)
    # frame_shapes_dict.append(shapes_dict)
    foreground_frames.append(foreground_frame)
    background_frames.append(background_frame)

    cv.imshow('frame', foreground_frame)
    cv.setWindowProperty('frame', cv.WND_PROP_TOPMOST, 1)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    video.write(foreground_frame)

# new approach directly compare frames and take average
video.release()
cv.destroyAllWindows()