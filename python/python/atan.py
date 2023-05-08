#!/usr/bin/python3

# SPDX-License-Identifier: MIT
# Copyright 2022 Eileen Yoon <eyn@gmx.com>

import ane
import numpy as np

model = ane.Model("atan.anec.so")

x = np.random.random((1, 1, 1, 4)).astype(np.float16)
pred = model.predict([x])
y = pred[0]
ref = np.arctan(x)
print(y, ref)
