// SPDX-License-Identifier: MIT
/* Copyright 2022 Eileen Yoon <eyn@gmx.com> */

#include <iostream>

#include "ane.h"
#include "anec_srgan.h"
#include "d4.h"

// cpp demo using D4

int main(void)
{
	int err = 0;

	struct ane_nn *nn = ane_init_srgan();
	if (nn == NULL) {
		std::cout << "failed to init" << std::endl;
		return -1;
	}

	auto input = D4::Tensor<uint16_t, 1, 3, 512, 512, D4_ALLOC_HEAP>::Constant(0x3c00);
	auto output = D4::Tensor<uint16_t, 1, 3, 2048, 2048, D4_ALLOC_HEAP>::Zero();

	ane_send_chan(nn, input.data(), 0);
	err = ane_exec(nn);
	if (err) {
		std::cout << "fuck" << std::endl;
	}
	else {
		printf("before: x: 0x%x\n", *(uint16_t *)(output.data()));
		ane_read_chan(nn, (void *)(output.data()), 0);
		printf("after: x: 0x%x\n", *(uint16_t *)(output.data()));
	}

	// output.to_file("out.bin");
	ane_free(nn);

	return err;
}
