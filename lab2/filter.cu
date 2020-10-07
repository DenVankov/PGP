#include "filter.h"

__constant__ int g_filter[6];

//Gray converter
__device__ double filterGrayScale(Pixel* pixel) {
    return pixel->x * 0.299 + pixel->y * 0.587 + pixel->z * 0.114;
}

__global__ void filterPrevittKernel(Pixel* pixel, int width, int height) {
    int idY = blockIdx.y * blockDim.y + threadIdx.y;
    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetY = gridDim.y * blockDim.y;
    int offsetX = gridDim.x * blockDim.x;

    for (int i = idY; i < height; i += offsetY) {
        for (int j = idX; j < width; j += offsetX) {
            double gy = 0.0;
            double gx = 0.0;
            Pixel onePixel;

            for (int k = 0; k < 3; ++k) {
                int row = i + k - 1;
                int row_0 = i - 1;
                int row_1 = i + 1;
                int col = j + k - 1;
                int col_0 = j - 1;
                int col_1 = j + 1;

                onePixel = tex2D(g_tex, col_0, row);
                gx += g_filter[k] * filterGrayScale(&onePixel);
                onePixel = tex2D(g_tex, col_1, row);
                gx += g_filter[k + 3] * filterGrayScale(&onePixel);
                onePixel = tex2D(g_tex, col, row_0);
                gy += g_filter[k] * filterGrayScale(&onePixel);
                onePixel = tex2D(g_tex, col, row_1);
                gy += g_filter[k + 3] * filterGrayScale(&onePixel);
            }

            Byte gm = (Byte)min((int)sqrt(gx * gx + gy * gy), (int)0xFF);

            int offset = i * width + j;
			pixel[offset].x = gm;
			pixel[offset].y = gm;
			pixel[offset].z = gm;
			pixel[offset].w = 0;
        }
    }
}
