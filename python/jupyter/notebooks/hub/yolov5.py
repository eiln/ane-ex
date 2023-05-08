#!/usr/bin/python3

# SPDX-License-Identifier: MIT
# Copyright 2022 Eileen Yoon <eyn@gmx.com>

import ane

import numpy as np
import cv2

def normalize(data): # https://stackoverflow.com/a/55141403/20891128
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class YOLOV5(ane.Model):
	def __init__(self, path):
		super(YOLOV5, self).__init__(path)

	def preprocess(self, img):
		# (any, any, 3) cv2 RGB -> (1, 3, 576, 576) inarr
		resized = cv2.resize(img, (576, 576), interpolation=cv2.INTER_AREA)
		transposed = np.expand_dims(resized.swapaxes(0, -1).swapaxes(1, 2), 0)
		normed = normalize(transposed).astype(np.float16)
		return normed

	def postprocess(self, outarrs):
		arr = np.vstack([outarrs[0].squeeze(), outarrs[1].squeeze(), outarrs[2].squeeze()])
		return arr
