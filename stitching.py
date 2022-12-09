"""
This file is derived from stitching sample from https://raw.githubusercontent.com/opencv/opencv/4.x/samples/python/stitching_detailed.py
"""

# Python 2/3 compatibility
from __future__ import print_function

import argparse
from collections import OrderedDict

import cv2 as cv
import numpy as np
from sqlalchemy import true

def get_matcher():
    try_cuda = False
    match_conf = 0.3 # 0.3 for orb, 0.65 for sift
    #range_width = 2
    matcher = cv.detail_BestOf2NearestMatcher(try_cuda, match_conf)
    #matcher = cv.detail_BestOf2NearestRangeMatcher(range_width, try_cuda, match_conf)
    return matcher


def get_compensator():
    compensator = cv.detail.ExposureCompensator_createDefault(cv.detail.ExposureCompensator_NO)
    return compensator

def stitch(full_images, image_masks, side_processing, prefix):
    work_megapix = 0.6 # Default
    seam_megapix = 0.1 # Default
    # work_megapix = 1.0 # Default
    # seam_megapix = 0.5 # Default
    compose_megapix = -1 # Default
    conf_thresh = 1.1 # Default
    ba_refine_mask = 'xxxxx' # Default
    wave_correct = cv.detail.WAVE_CORRECT_HORIZ # Default
    warp_type = 'spherical' # Default
    #blend_type = 'multiband' # Default
    blend_type = 'feather'
    blend_strength = 5 # [0,100] default 5
    timelapse_type = cv.detail.Timelapser_AS_IS
    finder = cv.ORB.create()
    seam_work_aspect = 1
    full_img_sizes = []
    features = []
    images = []
    is_work_scale_set = False
    is_seam_scale_set = False
    is_compose_scale_set = False
    idx = 0
    for full_img in full_images:
        full_img_sizes.append((full_img.shape[1], full_img.shape[0])) 
        if work_megapix < 0:
            img = full_img
            work_scale = 1
            is_work_scale_set = True
        else:
            if is_work_scale_set is False:
                work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_work_scale_set = True
            img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
        if is_seam_scale_set is False:
            if seam_megapix > 0:
                seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            else:
                seam_scale = 1.0
            seam_work_aspect = seam_scale / work_scale
            is_seam_scale_set = True
        img_feat = cv.detail.computeImageFeatures2(finder, img)
        features.append(img_feat)
        img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
        cv.imwrite(prefix + "input_" + str(idx) + ".jpg", img)
        idx += 1
        images.append(img)

    print("Resizing & compute features done")

    matcher = get_matcher()
    p = matcher.apply2(features)
    matcher.collectGarbage()

    # indices = cv.detail.leaveBiggestComponent(features, p, conf_thresh)
    # img_subset = []
    # full_img_sizes_subset = []
    # for i in range(len(indices)):
    #     img_subset.append(images[indices[i]])
    #     full_img_sizes_subset.append(full_img_sizes[indices[i]])
    # images = img_subset
    # full_img_sizes = full_img_sizes_subset
    num_images = len(images)
    print("Num effective images:", num_images)
    if num_images < 2:
        print("Need more images")
        exit()

    estimator = cv.detail_HomographyBasedEstimator() # Default
    b, cameras = estimator.apply(features, p, None)
    if not b:
        print("Homography estimation failed.")
        exit()
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    print("Homography estimation finished")

    adjuster = cv.detail_BundleAdjusterRay()
    #adjuster = cv.detail_BundleAdjusterReproj()
    adjuster.setConfThresh(conf_thresh)
    refine_mask = np.zeros((3, 3), np.uint8)
    if ba_refine_mask[0] == 'x':
        refine_mask[0, 0] = 1
    if ba_refine_mask[1] == 'x':
        refine_mask[0, 1] = 1
    if ba_refine_mask[2] == 'x':
        refine_mask[0, 2] = 1
    if ba_refine_mask[3] == 'x':
        refine_mask[1, 1] = 1
    if ba_refine_mask[4] == 'x':
        refine_mask[1, 2] = 1
    adjuster.setRefinementMask(refine_mask)
    #Debug only: check if there is an nan or inf value in giving matrix:
    
    
    b, cameras = adjuster.apply(features, p, cameras)

    if not b:
        print("Camera parameters adjusting failed.")
        exit()

    print("Camera parameters adjusting finished")

    focals = []
    for cam in cameras:
        focals.append(cam.focal)
    focals.sort()
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
    if wave_correct is not None:
        rmats = []
        for cam in cameras:
            rmats.append(np.copy(cam.R))
        rmats = cv.detail.waveCorrect(rmats, wave_correct)
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]
    corners = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []
    for i in range(0, num_images):
        um = cv.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
        masks.append(um)

    warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)  # warper could be nullptr?
    for idx in range(0, num_images):
        K = cameras[idx].K().astype(np.float32)
        swa = seam_work_aspect
        K[0, 0] *= swa
        K[0, 2] *= swa
        K[1, 1] *= swa
        K[1, 2] *= swa
        corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        corners.append(corner)
        sizes.append((image_wp.shape[1], image_wp.shape[0]))
        images_warped.append(image_wp)
        p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        masks_warped.append(mask_wp.get())

    images_warped_f = []
    for img in images_warped:
        imgf = img.astype(np.float32)
        images_warped_f.append(imgf)

    compensator = get_compensator()
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

    seam_finder = cv.detail_DpSeamFinder('COLOR') # Default
    masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)
    compose_scale = 1
    corners = []
    sizes = []
    blender = None
    timelapser = None
    result_images = []
    result_side = []
    # https://github.com/opencv/opencv/blob/4.x/samples/cpp/stitching_detailed.cpp#L725 ?
    for idx, full_img in enumerate(full_images):
        if not is_compose_scale_set:
            if compose_megapix > 0:
                compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            is_compose_scale_set = True
            compose_work_aspect = compose_scale / work_scale
            warped_image_scale *= compose_work_aspect
            warper = cv.PyRotationWarper(warp_type, warped_image_scale)
            for i in range(0, len(full_images)):
                cameras[i].focal *= compose_work_aspect
                cameras[i].ppx *= compose_work_aspect
                cameras[i].ppy *= compose_work_aspect
                sz = (int(round(full_img_sizes[i][0] * compose_scale)),
                      int(round(full_img_sizes[i][1] * compose_scale)))
                K = cameras[i].K().astype(np.float32)
                roi = warper.warpRoi(sz, K, cameras[i].R)
                corners.append(roi[0:2])
                sizes.append(roi[2:4])
        img = full_img
        K = cameras[idx].K().astype(np.float32)
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        
        # Use input mask
        if (image_masks is None):
            print("aaaaaaaaaa")
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        else:
            print("bbbbbbbbbbb")
            mask = image_masks[idx]

        p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        compensator.apply(idx, corners[idx], image_warped, mask_warped)
        image_warped_s = image_warped.astype(np.int16)
        dilated_mask = cv.dilate(masks_warped[idx], None)
        seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
        print(cv.UMat.get(seam_mask).shape,"hereyo:", mask_warped.shape)
        mask_warped = cv.bitwise_and(seam_mask, mask_warped)
        if blender is None:
            blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
            blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            if blend_width < 1:
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            elif blend_type == "multiband":
                blender = cv.detail_MultiBandBlender()
                blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
                #blender.setNumBands(0)
            elif blend_type == "feather":
                blender = cv.detail_FeatherBlender()
                blender.setSharpness(1. / blend_width)
            blender.prepare(dst_sz)
        if timelapser is None:
            timelapser = cv.detail.Timelapser_createDefault(timelapse_type)
            timelapser.initialize(corners, sizes)

        # Create both blender and timelapse
        ma_tones = np.ones((image_warped_s.shape[0], image_warped_s.shape[1]), np.uint8)
        timelapser.process(image_warped_s, ma_tones, corners[idx])
        fixed_file_name = prefix + str(idx) + ".jpg"
        result_image = timelapser.getDst()
        result_images.append(cv.UMat.get(result_image).copy())
        #cv.imwrite(fixed_file_name, result_image)
        blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])

        if side_processing is not None:
            print("shouldnt come here")
            side_set = side_processing[idx]
            processed_side_set = []
            idx2 = 0
            for side_image in side_set:
                # processed = image
                _, side_warped = warper.warp(side_image, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
                #compensator.apply(idx, corners[idx], side_warped, mask_warped)
                side_warped_s = side_warped.astype(np.int16)
                timelapser.process(side_warped_s, ma_tones, corners[idx])
                processed = cv.UMat.get(timelapser.getDst()).copy()
                processed_side_set.append(processed)
                #cv.imwrite(prefix + "side" + str(idx) + "_" + str(idx2) + ".jpg", processed)
                idx2 += 1
            result_side.append(processed_side_set)

    result = None
    result_mask = None
    result, result_mask = blender.blend(result, result_mask)
    result = u8image(result)
    #cv.imwrite(prefix + "blended.jpg", result)
    #cv.imwrite(prefix + "mask.jpg", result_mask)

    return result, result_mask, result_images, result_side

def u8image(img):
    if img.max() == 0:
        return np.uint8(img)
    img = img - img.min() # Now between 0 and 8674
    img = img / img.max() * 255
    return np.uint8(img)




def main():
    img_names = ['boat1.jpg', 'boat2.jpg', 'boat3.jpg', 'boat4.jpg', 'boat5.jpg', 'boat6.jpg']
    print(img_names)
    full_images = []
    for name in img_names:
        full_img = cv.imread(cv.samples.findFile(name))
        if full_img is None:
            print("Cannot read image ", name)
            exit()
        full_images.append(full_img)
    side_image = []

    images_12 = []
    images_12.append(full_images[0])
    images_12.append(full_images[1])
    images_12_blended, images_12_aligned, _ = stitch(images_12, None, "temp/12_")
    side_image.append(images_12_aligned)

    images_34 = []
    images_34.append(full_images[2])
    images_34.append(full_images[3])
    images_34_blended, images_34_aligned, _ = stitch(images_34, None, "temp/34_")
    side_image.append(images_34_aligned)

    images_56 = []
    images_56.append(full_images[4])
    images_56.append(full_images[5])
    images_56_blended, images_56_aligned, _ = stitch(images_56, None, "temp/56_")
    side_image.append(images_56_aligned)

    images_finals = []
    #images_finals.append(cv.imread(cv.samples.findFile("temp/12_blended.jpg")))
    #images_finals.append(cv.imread(cv.samples.findFile("temp/34_blended.jpg")))
    #images_finals.append(cv.imread(cv.samples.findFile("temp/56_blended.jpg")))
    images_finals.append(images_12_blended)
    images_finals.append(images_34_blended)
    images_finals.append(images_56_blended)
    image_blended, images_aligned, side_aligned = stitch(images_finals, side_image, "temp/final_")


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()