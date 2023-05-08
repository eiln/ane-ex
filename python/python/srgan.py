#!/usr/bin/python3

# SPDX-License-Identifier: MIT
# Copyright 2022 Eileen Yoon <eyn@gmx.com>

import ane
import argparse
import numpy as np
import cv2

def rescale(a, low, high):
    return np.interp(a, (a.min(), a.max()), (low, high))


class SRGAN(ane.Model):
	def __init__(self, path):
		super(SRGAN, self).__init__(path)

	def preprocess(self, img):
		# (any, any, 3) cv2 RGB -> (1, 3, 512, 512) inarr
		resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
		transposed = np.expand_dims(resized.swapaxes(0, -1).swapaxes(1, -1), 0)
		normed = rescale(transposed, -1, +1).astype(np.float16)
		return normed

	def postprocess(self, outarrs):
		# (1, 3, 2048, 2048) outarr -> (2048, 2048, 3) cv2 RGB
		reshaped = np.swapaxes(outarrs[0].squeeze(), 0, -1).swapaxes(0, 1)
		clipped = np.round(reshaped).clip(0, 255).astype(np.uint8)
		return clipped


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='srgan')
	parser.add_argument('-l', '--lib', help='anec dylib', default="srgan.anec.so")
	requiredNamed = parser.add_argument_group('required named arguments')
	requiredNamed.add_argument('-i', '--input', help='Input file name', required=True)
	requiredNamed.add_argument('-o', '--output', help='Output file name', required=True)
	args = parser.parse_args()

	model = SRGAN(args.lib)
	img = cv2.imread(args.input)[:,:,::-1]
	inarr = model.preprocess(img)
	outarrs = model.predict([inarr])
	pred = model.postprocess(outarrs)
	cv2.imwrite(args.output, pred[:,:,::-1])
