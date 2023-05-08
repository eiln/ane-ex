#!/usr/bin/python3

# SPDX-License-Identifier: MIT
# Copyright 2022 Eileen Yoon <eyn@gmx.com>

import ane

import numpy as np
import cv2

from torchvision import transforms
from PIL import Image


class FCN(ane.Model):
	def __init__(self, path):
		super(FCN, self).__init__(path)

	def preprocess(self, img):
		# (any, any, 3) cv2 RGB -> (1, 3, 640, 640) inarr
		resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
		trans = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		batch = trans(Image.fromarray(resized)).unsqueeze(0).numpy()
		return batch.astype(np.float16)

	def postprocess(self, outarrs):
		return outarrs[0]
