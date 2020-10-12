#include "gaussian.h"

using namespace std;

texture<uchar4, 2, cudaReadModeElementType> tex;

__constant__ double weights[1024 * 2 + 5];

__host__ double calc(int r) {
    double divisor = 0.0;
    double tmp_array[1024 * 2 + 5];
    for (int i = -r; i <= r; ++i) {
        double tmp = exp((-(double)(i * i)) / (double)(2 * r * r));
        divisor += tmp;
        tmp_array[i + r] = tmp;
    }
    CUDA_ERROR(cudaMemcpyToSymbol(weights, tmp_array, (1024*2 + 5) * sizeof(double)));

    return divisor;
}

__device__ __host__ double compare(int x, int side) {
    return (double)max(0, min(x, side));
}

__global__ void gaussian_filter(uchar4 *out_pixels, int w, int h, int r, bool flag, double divisor) {
    int idY = blockIdx.y * blockDim.y + threadIdx.y;
    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetY = gridDim.y * blockDim.y;
    int offsetX = gridDim.x * blockDim.x;

    uchar4 pixel;

    if (flag) {
        // Horisontal step
        for (int row = idY; row < h; row += offsetY) {
            for (int col = idX; col < w; col += offsetX) {
                double cur_func_res = 0.0;
                double out_r = 0.0, out_g = 0.0, out_b = 0.0;
                for (int k = - r; k <= r; ++k) {
                    //cur_func_res = exp((-(double)(k * k)) / (double)(2 * r * r));
                    cur_func_res = weights[k + r];

                    // Border case
                    double new_y = compare(row, h);
                    double new_x = compare(col + k, w);
                    // double new_y = max(0.0, (double)min(row, h)); // Because Horisontal
                    // double new_x = max(0.0, (double)min(col + k, w));

                    // Get pixel from texture
                    pixel = tex2D(tex, new_x, new_y);

                    // Calculate red, green, blue
                    out_r += pixel.x * cur_func_res;
                    out_g += pixel.y * cur_func_res;
                    out_b += pixel.z * cur_func_res;
                }
                int idx = col + row * w;
                out_pixels[idx].x = out_r / divisor;
                out_pixels[idx].y = out_g / divisor;
                out_pixels[idx].z = out_b / divisor;
                out_pixels[idx].w = 0.0;
            }
        }
    } else {
        // Vertical step
        for (int row = idY; row < h; row += offsetY) {
            for (int col = idX; col < w; col += offsetX) {
                double cur_func_res = 0.0;
                double out_r = 0.0, out_g = 0.0, out_b = 0.0;
                for (int k = - r; k <= r; ++k) {
                    //cur_func_res = exp((-(double)(k * k)) / (double)(2 * r * r));
                    cur_func_res = weights[k + r];
                    
                    // Border case
                    double new_y = compare(row + k, h);
                    double new_x = compare(col, w);
                    // double new_y = max(0.0, (double)min(row + k, h));
                    // double new_x = max(0.0, (double)min(col, w)); // Because Vertical

                    // Get pixel from texture
                    pixel = tex2D(tex, new_x, new_y);

                    // Calculate red, green, blue
                    out_r += pixel.x * cur_func_res;
                    out_g += pixel.y * cur_func_res;
                    out_b += pixel.z * cur_func_res;
                }
                int idx = col + row * w;
                out_pixels[idx].x = out_r / divisor;
                out_pixels[idx].y = out_g / divisor;
                out_pixels[idx].z = out_b / divisor;
                out_pixels[idx].w = 0.0;
            }
        }
    }
}

int main() {
    string nameIn;
    string nameOut;
    int r, w, h;

    cin >> nameIn;
    cin >> nameOut;
    cin >> r;

    FILE* in  = fopen(nameIn.c_str(), "rb");
    FILE* out = fopen(nameOut.c_str(), "wb");

    fread(&w, sizeof(int), 1, in);
    fread(&h, sizeof(int), 1, in);

    uchar4* pixels = (uchar4*)malloc(sizeof(uchar4) * w * h);

    fread(pixels, sizeof(uchar4), w * h, in);
    fclose(in);

    if (r <= 0) {
        fwrite(&w, sizeof(int), 1, out);
        fwrite(&h, sizeof(int), 1, out);
        fwrite(pixels, sizeof(uchar4), w * h, out);
        fclose(out);

        free(pixels);
        return 0;
    } else {
        double divisor = calc(r);

        cudaArray *array;
        cudaChannelFormatDesc channel = cudaCreateChannelDesc<uchar4>();
        CUDA_ERROR(cudaMallocArray(&array, &channel, w, h));
        CUDA_ERROR(cudaMemcpyToArray(array, 0, 0, pixels, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

        // Texture link preparing
        tex.channelDesc = channel;
    	tex.addressMode[0] = cudaAddressModeClamp;
    	tex.addressMode[1] = cudaAddressModeClamp;
    	tex.filterMode = cudaFilterModePoint;
    	tex.normalized = false;

        // Bind texture
        CUDA_ERROR(cudaBindTextureToArray(tex, array, channel));

        uchar4* out_pixels;
        CUDA_ERROR(cudaMalloc(&out_pixels, sizeof(uchar4) * w * h));

        // Horisontal, first step
        gaussian_filter<<<dim3(32, 32), dim3(32, 32)>>>(out_pixels, w, h, r, true, divisor);
        CUDA_ERROR(cudaGetLastError());
        CUDA_ERROR(cudaMemcpy(pixels, out_pixels, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

        // Vertical, second step
        CUDA_ERROR(cudaMemcpyToArray(array, 0, 0, pixels, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

        gaussian_filter<<<dim3(32, 32), dim3(32, 32)>>>(out_pixels, w, h, r, false, divisor);
        CUDA_ERROR(cudaGetLastError());
        CUDA_ERROR(cudaMemcpy(pixels, out_pixels, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

        // Unbind texture
        CUDA_ERROR(cudaUnbindTexture(tex));

        CUDA_ERROR(cudaFreeArray(array));
        CUDA_ERROR(cudaFree(out_pixels));

        fwrite(&w, sizeof(int), 1, out);
        fwrite(&h, sizeof(int), 1, out);
        fwrite(pixels, sizeof(uchar4), w * h, out);
        fclose(out);

        free(pixels);
    }

    return 0;
}
