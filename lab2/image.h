#ifndef PGP_LAB2_IMAGE_H
#define PGP_LAB2_IMAGE_H

#include <stdio.h>
#include <stdlib.h>

typedef const char* CString;
typedef uchar4 Pixel;
typedef unsigned char Byte;
typedef texture<Pixel, 2, cudaReadModeElementType> Texture2D;

typedef struct {
    int width;
    int height;
    Pixel* pixel;
} _image;

#define ERROR(call) { \
	cudaError_t err = call; \
	\
	if (err != cudaSuccess) { \
		fprintf(stderr, "ERROR: CUDA failed in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		exit(0); \
	} \
}


void createImage(_image* img, int width, int height);
void deleteImage(_image* img);
void readImageFromFile(_image* img, CString fileName);
void writeImageToFile(_image* img, CString fileName);
void createImageTexture(_image* img);
void deleteImageTexture();

extern cudaArray* g_arr;
extern Texture2D g_tex;

#endif //PGP_LAB2_IMAGE_H
