from collections import defaultdict
import numpy as np
import cv2 as cv
import datetime


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

def get_foreground_background_frames(play_video_window=False):
    # change args to use preset values for different videos
    # args = ["SAL_490_270_437", 1.1, 5, 0, 4, [2, 2, 2, 2]]
    # args = ["Stairs_490_270_346", 0.4, 15, 0, 8, [5, 5, 0, 0]]
    # args = ["test1_480_270_404", 0.6, 45, 0.1, 1, [5, 1, 9,12]]
    # args = ["test2_480_270_631", 0.7, 25, 0.1, 1, [6, 1, 0, 3]]
    args = ["test3_480_270_595", 0.6, 45, 0.2, 6, [5, 3, 6, 6]]
    foldername = args[0]
    # ang_threshold is the max diff allowed between any foreground vector angle and the common background angle
    ang_threshold = args[1]
    # nearest_frames is the number of frames used to average the moving object shapes to get a more stable object throughout the video
    nearest_frames = args[2]
    # quantile_threshold is used with nearest_frames variable, if the averaged value is above the threshold, then it is considered foreground
    quantile_threshold = args[3]
    # any moving block smaller than min_shape_threshold will be ignored and considered background
    min_shape_threshold = args[4]
    # any motion detected near the edges of the frame will be ignored according to the edge_buffers
    # the array includes the number of macroblocks to be ignored at four edges of the frame
    # edge buffer order: top down left right
    edge_buffers = args[5]

    print("Program started")

    imgs = []

    path_prefix = "video_files/"

    nums = foldername.split("_")
    width = int(nums[1])
    height = int(nums[2])
    frames = int(nums[3])
    path = path_prefix + foldername + "/"
    for i in range(frames):
        filename = foldername + "." + "{:03d}".format(i + 1) + ".rgb"
        imgs.append(readRGB(path, filename, width, height))

    frame1 = imgs[0]
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    foreground_frames = []
    background_frames = []
    frame_masks = []
    object_shapes = []
    final_frame_masks = []
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
        # this is the cv library that computes the optical flow(motion vectors) between two frames
        motion = cv.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 16, 3, 5, 0.5, 1)
        size = 16
        subsampled_motion = subsample(motion, size)
        subsampled_mag, subsampled_ang = cv.cartToPolar(subsampled_motion[..., 0], subsampled_motion[..., 1],
                                                        angleInDegrees=True)
        # the motion angles extracted from the vectors are values in 360 degrees
        # dividing them by 72 partitions them into 5 segments for finding the background(most common) motion angle
        subsampled_ang = subsampled_ang / 72
        m, n = subsampled_ang.shape
        u, c = np.unique(subsampled_ang.round(), return_counts=True)
        most_common_ang = u[c.argmax()]
        mask = np.zeros_like(subsampled_ang)

        # this loop filters out foreground objects by comparing against the most common angle
        # it also uses the edge_buffers to ignore certain areas near the edge of the frames
        for x in range(edge_buffers[0], m-edge_buffers[1]):
            for y in range(edge_buffers[2], n-edge_buffers[3]):
                if min(abs(subsampled_ang[x][y] - most_common_ang),
                       abs(5 - subsampled_ang[x][y] - most_common_ang)) > ang_threshold:
                    mask[x][y] = 1
        frame_masks.append(mask)
        prvs = nxt

    video = cv.VideoWriter('background_video.mp4',cv.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

    for i in range(len(imgs)-1):
        matrix = np.mean(frame_masks[max(0, i - nearest_frames):i + nearest_frames], axis=0)
        total_shapes = 0
        shapes = {}
        m, n = frame_masks[i].shape
        final_frame_mask = np.zeros((m,n))

        # this loop uses bfs to find all contiguous shapes and discards small moving segments
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
                            final_frame_mask[u][v] = 1
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
                        shapes[total_shapes] = visited
                        total_shapes += 1

        # this step upscales the foreground mask and produces the foreground object frame and the background frame with objected removed
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

        foreground_frames.append(foreground_frame)
        background_frames.append(background_frame)
        final_frame_masks.append(final_frame_mask)
        object_shapes.append(shapes)
        if play_video_window:
            cv.imshow('frame', foreground_frame)
            cv.setWindowProperty('frame', cv.WND_PROP_TOPMOST, 1)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
        video.write(background_frame)

    video.release()
    print("finished segmentation\n")
    return foreground_frames, background_frames, object_shapes, final_frame_masks


if __name__ == '__main__':
    # play_video_window is set to True here to display video as rendering for debug purposes
    foreground_frames, background_frames, object_shapes, final_frame_masks = get_foreground_background_frames(True)
