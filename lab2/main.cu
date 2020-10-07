#include "filter.h"

int main () {
    char fileIn[256];
    char fileOut[256];
    _image imageIn;
    _image imageOut;
    Pixel* pixel;
    int filter[] = {-1, -1, -1, 1, 1, 1};

    scanf("%s", fileIn);
    scanf("%s", fileOut);
    // printf("parced\n");
    //Creating out image and texture
    readImageFromFile(&imageIn, fileIn);
    int w = imageIn.width;
    int h = imageIn.height;
    createImage(&imageOut, w, h);
    createImageTexture(&imageIn);

    // printf("created\n");

    int n = sizeof(Pixel) * w * h;

    //Allocating
    ERROR(cudaMalloc(&pixel, n));
	ERROR(cudaMemcpyToSymbol(g_filter, filter, sizeof(filter)));

    dim3 gridSize(32, 32);
	dim3 blockSize(32, 32);

    // printf("allocated\n");
    filterPrevittKernel<<<gridSize, blockSize>>> (pixel, w, h);

    // printf("proceed\n");
    ERROR(cudaMemcpy(imageOut.pixel, pixel, n, cudaMemcpyDeviceToHost));
	ERROR(cudaFree(pixel));

    writeImageToFile(&imageOut, fileOut);
    deleteImageTexture();
    deleteImage(&imageIn);
    deleteImage(&imageOut);
    return 0;
}
