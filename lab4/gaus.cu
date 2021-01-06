// #include <bits/stdc++.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<vector>
#include<algorithm>
#include <climits>
// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/swap.h>
#include <thrust/extrema.h>
// #include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>

using namespace std;

typedef long long ll;

#define CUDA_ERROR(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "ERROR: CUDA failed in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		return(1); \
    } \
} \

struct abs_functor : public thrust::unary_function<double, double> {
    __host__ __device__ double operator()(double x) const {
        return x > 0.0 ? x : -x;
    }
};

struct abs_comparator{
    abs_functor fabs;
    __host__ __device__ double operator()(double x, double y){
        return fabs(x) < fabs(y);
    }
};

__global__ void kernel_compute_L (double* data, int n, int i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetx = gridDim.x * blockDim.x;
    for (int j = idx + i + 1; j < n; j += offsetx) {
        int q = 0;
        while (q < i) {
            data[j + i * n] -= data[j + q * n] * data[q + i * n];
            q++;
        }
        data[j + i * n] /= data[i + i * n];
    }
}

__global__ void kernel_compute_U (double* data, int n, int i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetx = gridDim.x * blockDim.x;
    for (int j = idx + i; j < n; j += offsetx) {
        int q = 0;
        while (q < i) {
            data[i + j * n] -= data[i + q * n] * data[q + j * n];
            q++;
        }
    }
}

__global__ void swapping(double* data, int n, int i, int max_id){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetx = gridDim.x * blockDim.x;

    for(int j = idx; j < n; j += offsetx){
        double tmp = data[i + j * n];
        data[i + j * n] = data[max_id + j * n];
        data[max_id + j * n] = tmp;
    }
}

int main(int argc, char const *argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    int n;
    cin >> n;

    vector<double> data(n * n);
    vector<int> p(n);

    for (int i = 0; i < n; ++i) {
        p[i] = i;

        for (int j = 0; j < n; ++j) {
            cin >> data[i + j * n];
        }
    }

    // LUP_decomp
    for (int i = 0; i < n; ++i) {
        int mx = INT_MIN;
        int idx = i;
        for (int j = i; j < n; ++j) {
            int uii = data[j + i * n];
            int q = 0;
            while (q < i) {
                uii -= data[j + q * n] * data[q + j * n];
                q++;
            }
            if (abs(uii) > mx) {
                mx = abs(uii);
                idx = j;
            }
        }

        p[i] = idx;
        for (int j = 0; j < n; ++j) {
                swap(data[i + j * n], data[idx + j * n]);
        }

        // Got U
        for (int j = i; j < n; ++j) {
            int q = 0;
            while (q < i) {
                data[i + j * n] -= data[i + q * n] * data[q + j * n];
                q++;
            }
        }

        // Got L
        for (int j = i + 1; j < n; ++j) {
            int q = 0;
            while (q < i) {
                data[j + i * n] -= data[j + q * n] * data[q + i * n];
                q++;
            }
            data[j + i * n] /= data[i + i * n];
        }
    }

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
    return 0;
}
