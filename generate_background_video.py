"""
Credit to juliachengw. As of commit cfe0b4b17bef617ff9c7d70be09f2a2983be1749 @11/23/22

The code is edited to generate video with foreground objected replaced with background
"""
import numpy as np
import cv2 as cv # version 4.6.0
# import matplotlib.pyplot as plt
import sys
from foreground_subtraction import get_foreground_background_frames
import datetime
from aligner import blending
from stitching import u8image

# min number of match points
MIN_MATCH_COUNT = 5
# the cadence of frames we use to generate panorama
N = 1
# min missing pixel count we use in the filling missing pixel processs. We stop searching if the missing pixel count is less than this number.
MIN_MISSING_PIXEL_COUNT = 10

# used to stop while loop earlier for faster processing
STOP_SEARCH_COUNT = 250


def output_frames(frame_list):
    video = cv.VideoWriter(str(datetime.datetime.now().hour)+ "_" + str(datetime.datetime.now().minute)+"_"+str(N)+'filled_video.mp4',cv.VideoWriter_fourcc(*'mp4v'), 20.0, (len(frame_list[0][0]), len(frame_list[0])))
    for frame in frame_list:
        video.write(frame)
    video.release()



def get_homography_matrices(background_frames):
    """
    generate a list of homography matrices that transforms frame n + N --> frame n for every Nth frame in the background
    :return: list of H matrices
    """
    homography_matrices = []
    for i in range(0, len(background_frames) - N, N):
        source_frame = background_frames[i + N]
        destination_frame = background_frames[i]
        # step 1. uses opencv's sift matching to find matching points between two frames
        # source - https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        # https://www.youtube.com/watch?v=6oLRdnQI_2w
        # convert image to grayscale to speed up processing
        destination_frame_gray = cv.cvtColor(destination_frame, cv.COLOR_BGR2GRAY)
        source_frame_gray = cv.cvtColor(source_frame, cv.COLOR_BGR2GRAY)
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(source_frame_gray, None) # kp1 is the source frame
        kp2, des2 = sift.detectAndCompute(destination_frame_gray, None) # kp2 is the destination frame
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=3)
        # Apply ratio test
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m in matches:
            if (m[0].distance < 0.6 * m[1].distance):
                good.append(m)
        matches = np.asarray(good)

        # step 2. get the homography matrix based on sift points
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
            H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            homography_matrices.append(H)

        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))

    return homography_matrices

def fill_missing_pixels_given_source_destination_frames(destination_frame, destination_frame_index, source_frame, source_frame_index, homography_matrices, inverse_homography_matrices, shapes, img, size=16):
    """
    helper method of fill_missing_pixels. fill the hole of the destination frame given a source frame
    """
    if source_frame_index > destination_frame_index:
        H = homography_matrices[source_frame_index // N - 1]
        for i in range(source_frame_index // N - 2, destination_frame_index // N - 1, -1):
            H = np.matmul(H, homography_matrices[i])
    else:
        H = inverse_homography_matrices[source_frame_index // N]
        for i in range(source_frame_index // N + 1, destination_frame_index // N):
            H = np.matmul(H, inverse_homography_matrices[i])
    # stitch images
    warped_source_frame = cv.warpPerspective(source_frame, H, (source_frame.shape[1], source_frame.shape[0]))
    warped_img_frame = cv.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    # fill holes in the destination image
    filled_destination_frame = np.copy(destination_frame)
    # tic = time.perf_counter()

    for shape in shapes.values():
            for x, y in shape:
                if x*(size+1) < len(warped_source_frame) and y*(size+1) < len(warped_source_frame[0]):
                    filled_destination_frame[x * size:(x + 1) * size, y * size:(y + 1) * size] = warped_source_frame[x * size:(x + 1) * size, y * size:(y + 1) * size]
    missing_pixel_count = np.count_nonzero(np.all(filled_destination_frame == [0, 0, 0], axis=2))
    return warped_img_frame, missing_pixel_count


total = 0

def fill_missing_pixels(curr_frame, curr_frame_index, homography_matrices, inverse_homography_matrices, background_frames, shapes, frame_mask, imgs):
    global total
    """
    fill holes of the given frame
    :param curr_frame: rgb value of the background frame
    :param curr_frame_index: the index of the frame
    :param homography_matrices: list of homography_matrices at every Nth frame
    :return: background frame with no holes
    """
    missing_pixel_count = float('inf')
    min_missing_pixel_count, best_index = missing_pixel_count, curr_frame_index
    best_warped_frame = curr_frame

    # look for source frame after current frame
    source_frame_index = curr_frame_index # the index of the source frame that we use to fill holes in the current frame
    search_count = 0
    while search_count < STOP_SEARCH_COUNT and source_frame_index + N < len(background_frames) and missing_pixel_count > MIN_MISSING_PIXEL_COUNT:
        search_count += 1
        source_frame_index += N
        warped_frame, missing_pixel_count = fill_missing_pixels_given_source_destination_frames(
            curr_frame, curr_frame_index, background_frames[source_frame_index], source_frame_index, homography_matrices, inverse_homography_matrices, shapes, imgs[source_frame_index])
        if min_missing_pixel_count > missing_pixel_count:
            min_missing_pixel_count, best_warped_frame = missing_pixel_count, warped_frame

    # look for source frame before current frame
    source_frame_index = curr_frame_index
    search_count = 0
    while search_count < STOP_SEARCH_COUNT and source_frame_index - N >= 0 and missing_pixel_count > MIN_MISSING_PIXEL_COUNT:
        search_count += 1
        source_frame_index -= N
        warped_frame, missing_pixel_count = fill_missing_pixels_given_source_destination_frames(
            curr_frame, curr_frame_index, background_frames[source_frame_index], source_frame_index, homography_matrices, inverse_homography_matrices, shapes, imgs[source_frame_index])
        if min_missing_pixel_count > missing_pixel_count:
            min_missing_pixel_count, best_warped_frame = missing_pixel_count, warped_frame

    kernel = np.ones((5, 5), np.uint8)
    frame_mask = cv.erode(frame_mask, kernel, iterations=1)
    best_filled_frame = blending(imgs[curr_frame_index], best_warped_frame, frame_mask)

    return u8image(best_filled_frame)

def stitching(imgs):
    # stitch the panorama
    stitcher = cv.Stitcher.create(mode=0) # mode 0 is panorama
    status, pano = stitcher.stitch(imgs)
    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)
    cv.imwrite("panorama.jpg", pano)
    print("stitching completed successfully. panorama saved!")
    print('Done')


def get_inverse_homography_matrices(homography_matrices):
    inverse_homography_matrices = []
    for h in homography_matrices:
        inverse_h = np.linalg.inv(h)
        inverse_homography_matrices.append(inverse_h)
    return inverse_homography_matrices


def main():
    foreground_frames, background_frames, object_shapes, final_frame_masks, imgs = get_foreground_background_frames()
    homography_matrices = get_homography_matrices(background_frames)
    inverse_homography_matrices = get_inverse_homography_matrices(homography_matrices)
    filled_background_frames = []
    print("\nstart filling missing pixels")
    ln = len(background_frames)-N
    for i in range(0, ln):
        if i == ln// 4:
            print("Processing 25% done")
        elif i == ln // 2:
            print("Processing 50% done")
        elif i == ln // 4 * 3:
            print("Processing 75% done")
        filled_result = fill_missing_pixels(background_frames[i], i, homography_matrices, inverse_homography_matrices, background_frames, object_shapes[i], final_frame_masks[i], imgs)
        filled_background_frames.append(filled_result)
    print("finished filling missing pixels\n\nstart stitching ")
    output_frames(filled_background_frames)


if __name__ == '__main__':
    main()
