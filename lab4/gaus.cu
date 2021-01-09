// #include <bits/stdc++.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<vector>
#include<algorithm>
#include <climits>
#include <thrust/swap.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

typedef long long ll;

#define CUDA_ERROR(err) { \
	if (err != cudaSuccess) { \
		fprintf(stderr, "ERROR: CUDA failed in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		return(1); \
	} \
} \

struct abs_comparator{
	__host__ __device__ bool operator()(double x, double y) {
		return fabs(x) < fabs(y);
	}
};

__global__ void kernel_compute_L (double* data, int n, int i) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetX = gridDim.x * blockDim.x;
	for (int j = idx + i + 1; j < n; j += offsetX) {
		data[j + i * n] /= data[i + i * n];
	}
}

__global__ void kernel_compute_U (double* data, int n, int i) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetX = gridDim.x * blockDim.x;
	int offsetY = gridDim.y * blockDim.y;
	for (int j = idx + i + 1; j < n; j += offsetX) {
		for (int q = idy + i + 1; q < n; q += offsetY) {
			data[j + q * n] -= data[j + i * n] * data[i + q * n];
		}
	}
}

// for (int j = i; j < n; ++j) {
//  for (int q = 0; q < i; ++q) {
//      data[i + j * n] -= data[i + q * n] * data[q + j * n];
//  }
// }

__global__ void kernel_swap(double* data, int n, int i, int max_idx) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetX = gridDim.x * blockDim.x;

	for (int j = idx; j < n; j += offsetX) {
		thrust::swap(data[i + j * n], data[max_idx + j * n]);
	}
}

int main(int argc, char const *argv[]) {
	ios_base::sync_with_stdio(false);
	cin.tie(nullptr);
	cout.tie(nullptr);
	int n;
	cin >> n;

	double* data, *dev_data;
	CUDA_ERROR(cudaMalloc((void**)&dev_data, sizeof(double) * n * n));
	int* p = (int*)malloc(sizeof(int) * n);
	data = (double*)malloc(sizeof(double) * n * n);

	// Data inputing + p initializing
	for (int i = 0; i < n; ++i) {
		p[i] = i;

		for (int j = 0; j < n; ++j) {
			cin >> data[i + j * n];
		}
	}

   // cudaEvent_t start, stop;
   // float gpu_time = 0.0;
   // cudaEventCreate(&start);
   // cudaEventCreate(&stop);
   // cudaEventRecord(start, 0);

	CUDA_ERROR(cudaMemcpy(dev_data, data, sizeof(double) * n * n, cudaMemcpyHostToDevice));
	// fprintf(stderr, "Got data\n");

	dim3 BLOCKS(32, 32);
	dim3 THREADS(32, 32);

	int max_idx;
	// double mx;
	abs_comparator cmp;
	thrust::device_ptr<double> data_ptr;
	thrust::device_ptr<double> max_ptr;

	for (int i = 0; i < n - 1; ++i) {
		max_idx = i;

		// Find max in column from i to n (cast to data_ptr == start)
		data_ptr = thrust::device_pointer_cast(dev_data + i * n);
		// Pointer e.x. largest == data + 3
		max_ptr = thrust::max_element(data_ptr + i, data_ptr + n, cmp);

		max_idx = max_ptr - data_ptr;
		// mx = fabs(*max_ptr);
		// fprintf(stderr, "Find max idx=%d\n", max_idx);
		// fprintf(stderr, "MAX=%f\n", mx);

		p[i] = max_idx;
		if (max_idx != i) {
			kernel_swap<<<32, 32>>>(dev_data, n, i, max_idx);
			CUDA_ERROR(cudaGetLastError());
		}

		kernel_compute_L<<<32, 32>>>(dev_data, n, i);
		CUDA_ERROR(cudaGetLastError());
		CUDA_ERROR(cudaThreadSynchronize());

		kernel_compute_U<<<BLOCKS, THREADS>>>(dev_data, n, i);
		CUDA_ERROR(cudaGetLastError());
		CUDA_ERROR(cudaThreadSynchronize());

		fprintf(stderr, "Iter=%d\n", i);
	}
	CUDA_ERROR(cudaMemcpy(data, dev_data, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
	CUDA_ERROR(cudaFree(dev_data));

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			// cout << data[i + j * n] << " ";
			printf("%.10e ", data[i + j * n]);
		}
		printf("\n");
	}

	for (int j = 0; j < n; ++j) {
		// cout << p[j] << " ";
		printf("%d ", p[j]);
	}

	printf("\n");

	free(data);
	free(p);

	// cudaEventRecord(stop, 0);
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&gpu_time, start, stop);
	// fprintf(stderr, "Time %f\n", gpu_time);
	return 0;
}
