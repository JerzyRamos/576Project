#!/usr/bin/env python

'''
  source - https://docs.opencv.org/4.x/d2/d8d/classcv_1_1Stitcher.html
  https://github.com/opencv/opencv/blob/4.x/samples/python/stitching.py
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

import argparse
import sys

modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

parser = argparse.ArgumentParser(prog='stitching.py', description='Stitching sample.')
parser.add_argument('--mode',
    type = int, choices = modes, default = cv.Stitcher_PANORAMA,
    help = 'Determines configuration of stitcher. The default is `PANORAMA` (%d), '
         'mode suitable for creating photo panoramas. Option `SCANS` (%d) is suitable '
         'for stitching materials under affine transformation, such as scans.' % modes)
parser.add_argument('--output', default = 'result.jpg',
    help = 'Resulting image. The default is `result.jpg`.')
parser.add_argument('img', nargs='+', help = 'input images')

__doc__ += '\n' + parser.format_help()

def readRGB(path, filename, width, height):
    f = open(path+filename, "r")
    arr = np.fromfile(f, np.uint8).reshape(height, width, 3)
    arr[:, :, [2, 0]] = arr[:, :, [0, 2]]
    return arr

def main():
    args = parser.parse_args()

    # default parameter image reading, replaced with temporary rgb reading code
    # read input images
    # imgs = []
    # for img_name in args.img:
    #     img = cv.imread(cv.samples.findFile(img_name))
    #     if img is None:
    #         print("can't read image " + img_name)
    #         sys.exit(-1)
    #     imgs.append(img)

    imgs = []
    foldername = "SAL_490_270_437"
    nums = foldername.split("_")
    width = int(nums[1])
    height = int(nums[2])
    frames = int(nums[3])
    path = foldername+"/"
    for i in range(1, frames+1, 30):
        filename = foldername+"."+"{:03d}".format(i)+".rgb"
        imgs.append(readRGB(path, filename, width, height))

    # Tried using masks to ignore the moving subject, but it's only ignored during stiching and not the final blending
    # so the subject will still show in the final panorama.

    # blackout the moving subject
    # for i in range(70, 200):
    #     for j in range(80, 130):
    #         imgs[0][i, j, :] = 0

    # creating the masks
    # masks = []
    # for i in range(15):
    #     masks.append(np.zeros((height, width, 1), np.uint8))
    #     masks[i].fill(255)
    # for i in range(70, 200):
    #     for j in range(80, 130):
    #         masks[0][i, j, :] = 0

    print(len(imgs))
    stitcher = cv.Stitcher.create(args.mode)
    status, pano = stitcher.stitch(imgs)
    # status, pano = stitcher.stitch(imgs, masks) # if using masks

    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)

    cv.imwrite(args.output, pano)
    print("stitching completed successfully. %s saved!" % args.output)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()