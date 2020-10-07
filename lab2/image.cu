#include "image.h"

cudaArray* g_arr;
Texture2D g_tex;

void createImage(_image* img, int width, int height) {
	img->width  = width;
	img->height = height;
	img->pixel  = (Pixel*)malloc(sizeof(Pixel) * width * height);
}

void deleteImage(_image* img) {
	free(img->pixel);

	img->width = 0;
	img->height = 0;
	img->pixel = NULL;
}

void readImageFromFile(_image* img, CString fileName) {
	FILE* file = fopen(fileName, "rb");
	int width;
	int height;

	fread(&width, sizeof(width), 1, file);
	fread(&height, sizeof(height), 1, file);
	createImage(img, width, height);
    // printf("w: %d, h: %d\n", width, height);
	fread(img->pixel, sizeof(Pixel), img->width * img->height, file);
    // printf("%s\n", &(img->pixel));
    // printf("scanned\n");
	fclose(file);
}

void writeImageToFile(_image* img, CString fileName) {
	FILE* file = fopen(fileName, "wb");

	fwrite(&img->width, sizeof(img->width), 1, file);
	fwrite(&img->height, sizeof(img->height), 1, file);
	fwrite(img->pixel, sizeof(Pixel), img->width * img->height, file);
	fclose(file);
}

void createImageTexture(_image* img) {
	int w = img->width;
	int h = img->height;

	g_tex.channelDesc = cudaCreateChannelDesc<Pixel>();
	g_tex.addressMode[0] = cudaAddressModeClamp;
	g_tex.addressMode[1] = cudaAddressModeClamp;
	g_tex.filterMode = cudaFilterModePoint;
	g_tex.normalized = false;

	ERROR(cudaMallocArray(&g_arr, &g_tex.channelDesc, w, h));
	ERROR(cudaMemcpyToArray(g_arr, 0, 0, img->pixel, sizeof(Pixel) * w * h, cudaMemcpyHostToDevice));
	ERROR(cudaBindTextureToArray(g_tex, g_arr, g_tex.channelDesc));
}

void deleteImageTexture() {
	ERROR(cudaUnbindTexture(g_tex));
	ERROR(cudaFreeArray(g_arr));
}
