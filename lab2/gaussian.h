#ifndef PGP_LAB2_GAUSSIAN_H
#define PGP_LAB2_GAUSSIAN_H

#include <bits/stdc++.h>

#define CUDA_ERROR(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "ERROR: CUDA failed in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		return(1); \
    } \
} \

#endif // PGP_LAB2_GAUSSIAN_H
