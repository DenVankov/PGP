#ifndef PGP_LAB2_FILTER_H
#define PGP_LAB2_FILTER_H

#include "image.h"
#include <math.h>

__device__ double filterGrayScale(Pixel* pixel);
__global__ void filterPrevittKernel(Pixel* pixel, int width, int height);
__constant__ extern int g_filter[6];

#endif //PGP_LAB2_FILTER_H
