"""
"""
import math
from pickle import TRUE
import cv2 as cv
import numpy as np
from foreground_subtraction import get_foreground_background_frames
from stitching import stitch
# from generate_panorama02 import fill_missing_pixels, get_homography_matrices, get_inverse_homography_matrices

N = 68 # Number of images processed in one go


# adapted from https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html
# https://gist.github.com/royshil/0b20a7dd08961c83e210024a4a7d841a


def blending(A, B, m, num_levels=6):
    # assume mask is uint8 [0,255]
    m = np.float32(m)
    m = m / 255
    m = cv.cvtColor(m, cv.COLOR_GRAY2BGR)

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv.pyrDown(GA)
        GB = cv.pyrDown(GB)
        GM = cv.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpA = [gpA[num_levels-1]]
    lpB = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1, 0, -1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i-1], cv.resize(cv.pyrUp(gpA[i]),
                         (gpA[i-1].shape[1], gpA[i-1].shape[0])))
        LB = np.subtract(gpB[i-1], cv.resize(cv.pyrUp(gpB[i]),
                         (gpB[i-1].shape[1], gpB[i-1].shape[0])))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1])  # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, num_levels):
        ls_ = cv.resize(cv.pyrUp(ls_), (LS[i].shape[1], LS[i].shape[0]))
        ls_ = cv.add(ls_, LS[i])

    return ls_


def align(foreground_frames, background_frames, mask_frames):
    frames = len(background_frames)

    # 1st pass
    blended_images = []
    blended_masks = []
    side_images = []

    # for batch in range(math.ceil(frames / N)-1):
    #     print("Batch", batch)
    #     start_frame = batch * N
    #     end_frame = (batch + 1) * N
    #     if end_frame > frames:
    #         end_frame = frames
    #     input_batch = background_frames[start_frame: end_frame]
    #     input_mask = mask_frames[start_frame: end_frame]
    #     # input_batch = []
    #     # input_mask = []
    #     # Sides are applied with the same wrap/ compensation that's applied to stitched image
    #     # Use this to process foreground
    #     input_sides = []
    #     for i in range(start_frame, end_frame):
    #         # input_batch.append(foreground_frames[i])
    #         # input_mask.append(foreground_frames[i])
    #         foreground_img = foreground_frames[i]
    #         fg_mask = mask_frames[i].copy()
    #         fg_mask = cv.bitwise_not(fg_mask)
    #         # fg_mask = fg_mask.astype('float32')
    #         mask = cv.cvtColor(fg_mask, cv.COLOR_GRAY2BGR)
    #         input_sides.append([foreground_img, mask])
    #
    #     blended, mask, _, output_sides = stitch(
    #         input_batch, input_mask, input_sides, "temp/batch_1_" + str(batch) + "_")
    #     blended_images.append(blended)
    #     blended_masks.append(mask)
    #     # Aligned contains images alined to the final position
    #     # Also add flattened foreground output
    #     aligned = []
    #     for frame in output_sides:
    #         for image in frame:
    #             aligned.append(image)
    #     side_images.append(aligned)
    #
    # prefix = "test2_"
    # for i in range(len(blended_images)):
    #     cv.imwrite(prefix + "blended_"+str(i)+".jpg", blended_images[i])
    # for i in range(len(blended_masks)):
    #         cv.imwrite(prefix + "mask"+str(i)+".jpg", blended_masks[i])
    # for i in range(len(side_images)):
    #     for j in range(len(side_images[i])):
    #         cv.imwrite(prefix + "side_"+str(i)+"_"+str(j)+".jpg", side_images[i][j])

    prefix = "test2_"
    blended_images = []
    blended_masks = []
    for i in range(8):
        a = cv.imread(prefix + "blended_"+str(i)+".jpg")
        blended_images.append(a.astype(np.uint8))
        mask_img = cv.imread(prefix + "mask"+str(i)+".jpg")
        mask = np.zeros((len(mask_img), len(mask_img[0])))
        for x in range(len(mask_img)):
            for y in range(len(mask_img[0])):
                if mask_img[x][y].all() == 255:
                    mask[x][y] = 255
        blended_masks.append(mask.astype(np.uint8))


    # Read from files for testing
    # side_count = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 25]
    # for i in range(11):
    #     blended_images.append(cv.imread("test_inputs/final_input_" + str(i) + ".jpg"))
    #     mask = cv.imread("test_inputs/batch_1_" + str(i) + "_mask.jpg")
    #     mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    #     _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    #     blended_masks.append(mask)
    #     aligned = []
    #     for j in range(side_count[i]):
    #         aligned.append(cv.imread("test_inputs/batch_1_" + str(i) + "_side" + str(j) + "_0.jpg"))
    #     side_images.append(aligned)

    # # Fill Doesn't work, do not enable
    # homography_matrices = get_homography_matrices(blended_images)
    # inverse_homography_matrices = get_inverse_homography_matrices(homography_matrices)
    # filled_background_frames = []
    # print("\nstart filling missing pixels")
    # for i in range(len(blended_images)):
    #     filled_result = fill_missing_pixels(blended_images[i], i, homography_matrices, inverse_homography_matrices, blended_images)
    #     filled_background_frames.append(filled_result)
    #     cv.imwrite("temp/" + str(i) + ".jpg", filled_result)
    # print("Done")

    # 2nd pass
    blended, final_mask, final_aligned, side_aligned = stitch(
        blended_images, blended_masks, None, "temp/final_")

    # final_aligned = []
    # side_aligned = []
    # for i in range(11):
    #     final_aligned.append(cv.imread("test_inputs/final_" + str(i) + ".jpg"))
    # for i in range(11):
    #     aligned = []
    #     for j in range(side_count[i] * 2):
    #         aligned.append(cv.imread("test_inputs/final_side" + str(i) + "_" + str(j) + ".jpg"))
    #     side_aligned.append(aligned)

    # Flatten side image
    side_aligned_output = []
    side_aligned_mask = []
    for aligned in side_aligned:
        for i in range(len(aligned) // 2):
            side_aligned_output.append(aligned[i * 2])
            mask = aligned[i * 2 + 1].astype(np.uint8)
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            mask = cv.bitwise_not(mask)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv.dilate(mask, kernel, iterations=2)
            side_aligned_mask.append(mask)

    # Hacky way to recreate mask for now
    final_masks = []
    for i in range(7):
        mask = final_aligned[i].astype(np.uint8)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(mask, 5, 255, cv.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.dilate(mask, kernel, iterations=2)
        final_masks.append(mask)

    blended = final_aligned[0]
    for i in range(1,7):
        blended = blending(blended, final_aligned[i], final_masks[i], 2)
    cv.imwrite("output.jpg", blended)

    return blended, side_aligned_output, side_aligned_mask


def main():
    foreground_frames, background_frames,_,mask_frames = get_foreground_background_frames()
    align(foreground_frames, background_frames, mask_frames)


if __name__ == '__main__':
    main()