"""
Credit to juliachengw. As of commit cfe0b4b17bef617ff9c7d70be09f2a2983be1749 @11/23/22

The code is edited to generate a motion trail image for the subject
"""
import numpy as np
import cv2 as cv # version 4.6.0
# import matplotlib.pyplot as plt
import sys
from foreground_subtraction_motion_trial import get_foreground_background_frames
import datetime
from aligner import blending
# min number of match points
MIN_MATCH_COUNT = 4
# the cadence of frames we use to generate panorama
N = 1
# min missing pixel count we use in the filling missing pixel processs. We stop searching if the missing pixel count is less than this number.
MIN_MISSING_PIXEL_COUNT = 50

# used to stop while loop earlier for faster processing
STOP_SEARCH_COUNT = 180


def output_frames(frame_list):
    video = cv.VideoWriter("00pan"+str(datetime.datetime.now().hour)+ "_" + str(datetime.datetime.now().minute)+"_"+str(N)+'panfilled_video.mp4',cv.VideoWriter_fourcc(*'mp4v'), 20.0, (len(frame_list[0][0]), len(frame_list[0])))
    for frame in frame_list:
        video.write(frame)
    video.release()



def get_homography_matrices(background_frames, pano):
    """
    generate a list of homography matrices that transforms frame n + N --> frame n for every Nth frame in the background
    :return: list of H matrices
    """
    homography_matrices = []
    destination_frame = pano
    destination_frame_gray = cv.cvtColor(destination_frame, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp2, des2 = sift.detectAndCompute(destination_frame_gray, None)   # kp2 is the destination frame
    for i in range(0, len(background_frames) - N, N):
        source_frame = background_frames[i]
        # step 1. uses opencv's sift matching to find matching points between two frames
        # source - https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        # https://www.youtube.com/watch?v=6oLRdnQI_2w
        # convert image to grayscale to speed up processing
        source_frame_gray = cv.cvtColor(source_frame, cv.COLOR_BGR2GRAY)

        # Initiate SIFT detector

        # find the keypoints and descriptors with SIFT

        kp1, des1 = sift.detectAndCompute(source_frame_gray, None) # kp1 is the source frame
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m in matches:
            if (m[0].distance < 0.75 * m[1].distance):
                good.append(m[0])
        matches = np.asarray(good)

        # step 2. get the homography matrix based on sift points
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            homography_matrices.append(H)

        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    print("total len:", len(homography_matrices))
    return homography_matrices

def fill_missing_pixels_given_source_destination_frames(destination_frame, source_frame_index, source_frame, homography_matrices, mask, size=16):
    """
    helper method of fill_missing_pixels. fill the hole of the destination frame given a source frame
    """
    H = homography_matrices[source_frame_index // N]
    # stitch images
    warped_source_frame = cv.warpPerspective(source_frame, H, (destination_frame.shape[1], destination_frame.shape[0]))
    warped_mask = cv.warpPerspective(mask, H, (destination_frame.shape[1], destination_frame.shape[0]))
    kernel = np.ones((5, 5), np.uint8)
    warped_mask = cv.dilate(warped_mask, kernel, iterations=4)
    best_filled_frame = blending(warped_source_frame, destination_frame, warped_mask)
    cv.imwrite("video_files/matches_" + str(source_frame_index) + ".jpg", best_filled_frame)

    missing_pixel_count = 0
    return best_filled_frame, missing_pixel_count




def fill_missing_pixels(curr_frame, curr_frame_index, homography_matrices, background_frames, mask, pano):
    """
    fill holes of the given frame
    :param curr_frame: rgb value of the background frame
    :param curr_frame_index: the index of the frame
    :param homography_matrices: list of homography_matrices at every Nth frame
    :return: background frame with no holes
    """
    missing_pixel_count = float('inf')

    filled_frame, missing_pixel_count = fill_missing_pixels_given_source_destination_frames(
        pano, curr_frame_index, curr_frame, homography_matrices, mask)


    return np.asarray(filled_frame)

def stitching(imgs):
    # stitch the panorama
    stitcher = cv.Stitcher.create(mode=0) # mode 0 is panorama
    status, pano = stitcher.stitch(imgs)
    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)
    cv.imwrite("0000_nice_panorama.jpg", pano)
    print("stitching completed successfully. panorama saved!")
    print('Done')


def main():
    pano = cv.imread("video_files/panorama.jpg")
    foreground_frames, background_frames, object_shapes, final_frame_masks, imgs = get_foreground_background_frames()
    homography_matrices = get_homography_matrices(background_frames, pano)
    # inverse_homography_matrices = get_inverse_homography_matrices(homography_matrices)
    filled_background_frames = []
    print("\nstart filling missing pixels: it's going to take a while")
    ln = len(background_frames) - N
    Z = 48
    # preset_list = [0,48,96,144,246,362,425,480,528,576]
    # for i in preset_list:
    #     pano = fill_missing_pixels(imgs[i], i, homography_matrices, background_frames, final_frame_masks[i], pano)

    for i in range(0, ln, Z):
        if i == ln // Z // 4:
            print("Processing 25% done")
        elif i == ln // Z // 2:
            print("Processing 50% done")
        elif i == ln // Z // 4 * 3:
            print("Processing 75% done")
        pano = fill_missing_pixels(imgs[i], i, homography_matrices, background_frames, final_frame_masks[i], pano)

    cv.imwrite("00000pan_final"+str(datetime.datetime.now().hour) + "_" + str(datetime.datetime.now().minute)+".jpg", pano)
    print("finished filling missing pixels\n\nstart stitching: hold tight, it's going to take a long while")


if __name__ == '__main__':
    main()