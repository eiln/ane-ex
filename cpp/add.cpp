// SPDX-License-Identifier: MIT
/* Copyright 2022 Eileen Yoon <eyn@gmx.com> */

#define EIGEN_DEFAULT_TO_ROW_MAJOR

#include <iostream>
#include <Eigen/Dense>

#include "ane.h"
#include "ane_f16.h"
#include "anec_add.h"

// cpp demo using Eigen

int main(void)
{
	int err = 0;

	Eigen::MatrixXf mat1 = Eigen::MatrixXf::Constant(123, 456, 1.0);
	Eigen::MatrixXf mat2 = Eigen::MatrixXf::Constant(123, 456, 2.0);
	Eigen::MatrixXf mat4 = Eigen::MatrixXf::Constant(123, 456, 3.0);

	struct ane_nn *nn = ane_init_add();
	if (nn == NULL) {
		std::cout << "failed to init" << std::endl;
		return -1;
	}

	float_to_half_array(mat1.data(), dat1, mat1.size());
	float_to_half_array(mat2.data(), dat2, mat2.size());
	float_to_half_array(mat4.data(), dat4, mat4.size());
	init_half_array(dat3, mat4.size());

	for (int i = 0; i < 100; i ++){
		ane_send(nn, dat1, 0);
		ane_send(nn, dat2, 1);
		err = ane_exec(nn);
		if (err){
			std::cout << "fuck" << std::endl;
		}
		ane_read(nn, dat3, 0);

		if (memcmp(dat3, dat4, sizeof(dat3))) {
			std::cout << "fuck" << std::endl;
		}
	}

	ane_free(nn);

	return err;
}
