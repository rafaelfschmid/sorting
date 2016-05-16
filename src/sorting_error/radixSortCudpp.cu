/*
 ============================================================================
 Name        : sorting_segments.cu
 Author      : Rafael Schmid
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */

#include "cudpp.h"
#include <chrono>
#include <iostream>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

void cudaTest(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("cuda returned error %s (code %d), line(%d)\n",
				cudaGetErrorString(error), error, __LINE__);
		exit (EXIT_FAILURE);
	}
}

void print(int* host_data, int n) {
	std::cout << "\n";
	for (int i = 0; i < n; i++) {
		std::cout << host_data[i] << " ";
	}
	std::cout << "\n";
}

int main(void) {

	int num_of_elements;
	int i;

	scanf("%d", &num_of_elements);
	int mem_size_vec = sizeof(int) * num_of_elements;
	int *h_vec = (int *) malloc(mem_size_vec);
	int *h_value = (int *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++) {
		scanf("%d", &h_vec[i]);
		h_value[i] = i;
	}

	CUDPPHandle theCudpp;
	cudppCreate(&theCudpp);

	CUDPPConfiguration config;
	//config.op = CUDPP_OPERATOR_INVALID;
	config.datatype = CUDPP_INT;
	config.algorithm = CUDPP_SORT_RADIX;
	//config.options = CUDPP_OPTION_FORWARD;

	CUDPPHandle sortplan = 0;
	CUDPPResult res = cudppPlan(theCudpp, &sortplan, config, num_of_elements, 1,
			0);
	if (CUDPP_SUCCESS != res) {
		printf("Error creating CUDPPPlan\n");
		exit(-1);
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int *d_value, *d_vec;

	cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_value, mem_size_vec));

	cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));
	cudaTest(
			cudaMemcpy(d_value, h_value, mem_size_vec, cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	cudppRadixSort(sortplan, d_vec, d_value, num_of_elements);
	cudaEventRecord(stop);

	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess)
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

	cudaMemcpy(h_value, d_value, mem_size_vec, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vec, d_vec, mem_size_vec, cudaMemcpyDeviceToHost);

	if (ELAPSED_TIME == 1) {
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << milliseconds << "\n";
	} else
		print(h_vec, num_of_elements);

	free(h_vec);
	cudaFree(d_vec);
	cudaFree(d_value);
	return 0;
}
