// #include <bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
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

#define NUM_BLOCKS 8
#define BLOCK_SIZE 1024

__device__ void swap_step(int* nums, int* tmp, int size, int start, int stop, int step, int i) {
	// Using shared memory to store blocks and sort them
	__shared__ int sh_array[BLOCK_SIZE];

	// Step for bitonic merge inside merging
	for (int shift = start; shift < stop; shift += step) {
		// New start pointer
		ctmp = nums + shift;

		// Right side
		if (i >= BLOCK_SIZE / 2)
			sh_array[i] = tmp[BLOCK_SIZE * 3 / 2 - 1 - i];
		else
			sh_array[i] = tmp[i];

		__syncthreads();

		// From half
		for (int j = BLOCK_SIZE / 2; j > 0; j /= 2) {
			int XOR = i ^ j;
			// The threads with the lowest ids sort the array
			if (XOR > i) {
				if ((i & BLOCK_SIZE) != 0) {
					// Sort descending, swap(i, XOR)		
					if (sh_array[i] < sh_array[XOR])
						thrust::swap(sh_array[i], sh_array[XOR]);
				} else {
					// Sort ascending, swap(i, XOR)
					if (sh_array[i] > sh_array[XOR])
						thrust::swap(sh_array[i], sh_array[XOR]);
				}
			}

			__syncthreads();
		}

		// Back from shared to temporary
		tmp[i] = sh_array[i];
	}
}

__global__ void kernel_bitonic_merge_step(int* nums, int size, bool is_odd, bool flag) {
	// Temporary array for splitting into blocks
	int* tmp = nums;

	// Every thread gets exactly one value in the unsorted array
	int id_block = blockIdx.x;
	int offset = gridDim.x;
	int i = threadIdx.x;

	// For odd step
	if(is_odd) {
		swap_step(nums, tmp, size, (BLOCK_SIZE / 2) + id_block * BLOCK_SIZE, size - BLOCK_SIZE, offset * BLOCK_SIZE, i);
	} else { // For even step
		swap_step(nums, tmp, size, id_block * BLOCK_SIZE, size, offset * BLOCK_SIZE, i);
	}
}

__global__ void bitonic_sort_step(int *nums, int j, int k, int size) {
	// Using shared memory to store blocks and sort them
	__shared__ int sh_array[BLOCK_SIZE];

	// Temporary array for splitting into blocks
	int* tmp = nums;

	// Every thread gets exactly one value in the unsorted array
	int id_block = blockIdx.x;
	int offset = gridDim.x;
	int i = threadIdx.x;

	// Step for bitonic sort
  	for (int shift = id_block * BLOCK_SIZE; shift < size; shift += offset * BLOCK_SIZE) {
			// New start pointer
			tmp = nums + shift;

			// Store in shared memory
			sh_array[i] = tmp[i];

			__syncthreads();

			// From half
			for (j = k / 2; j > 0; j /= 2) {
				int XOR = i ^ j;
				// The threads with the lowest ids sort the array
				if (XOR > i) {
					if ((i & k) != 0) {
						// Sort descending, swap(i, XOR)		
						if (sh_array[i] < sh_array[XOR])
							thrust::swap(sh_array[i], sh_array[XOR]);
					} else {
						// Sort ascending, swap(i, XOR)
						if (sh_array[i] > sh_array[XOR])
							thrust::swap(sh_array[i], sh_array[XOR]);
					}
				}

				__syncthreads();
			}

			// Back from shared to temporary
			tmp[i] = sh_array[i];
		}
}

int main(int argc, char *argv[]) {
	ios_base::sync_with_stdio(false);
	cin.tie(nullptr);
	cout.tie(nullptr);
	int size, upd_size;

	// Allocating + inputting
	scanf("%d", &size);
	upd_size = ceil((double)size / BLOCK_SIZE) * BLOCK_SIZE;
	int* data = (int*)malloc(sizeof(int) * upd_size);
	int* dev_data;
	CUDA_ERROR(cudaMalloc((void**)&dev_data, sizeof(int) * upd_size));

	for (int i = 0; i < size; ++i) {
		// fread(&size, sizeof(int), 1, stdin);
		scanf("%d", &data[i]);
		// fprintf(stderr, "%d ", size);
	}

	for (int i = size; i < upd_size; ++i) {
		data[i] = INT_MAX;
	}

	// Copy to device
	CUDA_ERROR(cudaMemcpy(dev_data, data, upd_size, cudaMemcpyHostToDevice));
	
	////////////////////////////////////////////////////////////////////////////////////////
	// Pre sort of all blocks by bitonic sort
  	// Main step
  	for (int k = 2; k <= upd_size; k *= 2) {
    	// Merge and split step
    	for (int j = k / 2; j > 0; j /= 2) {
      		bitonic_sort_step<<<NUM_BLOCKS, BLOCK_SIZE>>>(dev_data, j, k, upd_size);
			CUDA_ERROR(cudaGetLastError());
    	}
  	}
  	////////////////////////////////////////////////////////////////////////////////////////

	/* Sort of buckets with bitonic merge inside
	| 1 3 5 7 | 2 4 6 8 | -> | 1 2 3 4 5 6 7 8| (size == 8)
	
	Including 2 steps merge + splitting
	*/
	for (int i = 0; i < 2 * (upd_size / BLOCK_SIZE); ++i) {
        kernel_bitonic_merge_step<<<NUM_BLOCKS, NUM_BLOCKS>>>(dev_data, upd_size, (bool)(i % 2), true);
    }

	CUDA_ERROR(cudaMemcpy(data, dev_data, upd_size, cudaMemcpyDeviceToHost))
	CUDA_ERROR(cudaFree(dev_data));

	for (int i = 0; i < size; ++i) {
		printf("%d ", data[i]);
	}
	printf("\n");

	free(data);
	return 0;
}
