#include "mpi.h"
// #include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

using namespace std;

#define left 0
#define right 1
#define front 2
#define back 3
#define down 4
#define up 5

#define on_x 0
#define on_y 1
#define on_z 2

const int NDIM = 3;
const int NDIM_2 = 6;

int id, ib, jb, kb;
// int dimensions[NDIM];
int npx, npy, npz;
int blocks[NDIM];
double l[NDIM];
double u[NDIM_2];
string filename;
double eps, u0;
double hx, hy, hz;

__constant__ int g_dimensions[3];

// Индексация внутри блока
#define _i(i, j, k) (((k) + 1) * (npy + 2) * (npx + 2) + ((j) + 1) * (npx + 2) + (i) + 1)
#define _iz(id) (((id) / (npx + 2) / (npy + 2)) - 1)
#define _iy(id) ((((id) % ((npx + 2) * (npy + 2))) / (npx + 2)) - 1)
#define _ix(id) ((id) % (npx + 2) - 1)

// Индексация по блокам (процессам)
#define _ib(i, j, k) ((k) * blocks[on_y] * blocks[on_x] + (j) * blocks[on_x] + (i))
#define _ibz(id) ((id) / blocks[on_x] / blocks[on_y])
#define _iby(id) (((id) % (blocks[on_x] * blocks[on_y])) / blocks[on_x])
#define _ibx(id) ((id) % blocks[on_x])

#define CUDA_ERROR(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "ERROR: CUDA failed in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return(1); \
    } \
} \

__device__ int _ind(int i, int j, int k) {
    return (((k) + 1) * (g_dimensions[1] + 2) * (g_dimensions[0] + 2) + ((j) + 1) * (g_dimensions[0] + 2) + (i) + 1);
}

__global__ void kernel_copy_edge_xy(double* edge_xy, double* data, int nx, int ny, int nz, int k, bool flag, double u) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;
    int i, j;
    if (flag) {
        for (i = idx; i < nx; i += offsetX)
            for (j = idy; j < ny; j += offsetY)
                edge_xy[i + j * nx] = data[_ind(i, j, k)];
    } else {
        if (edge_xy) {
            for (i = idx; i < nx; i += offsetX)
                for (j = idy; j < ny; j += offsetY)
                    data[_ind(i, j, k)] = edge_xy[i + j * nx];
        } else {
            for (i = idx; i < nx; i += offsetX)
                for (j = idy; j < ny; j += offsetY)
                    data[_ind(i, j, k)] = u;
        }
    }
}

__global__ void kernel_copy_edge_xz(double* edge_xz, double* data, int nx, int ny, int nz, int j, bool flag, double u) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;
    int i, k;
    if (flag) {
        for (i = idx; i < nx; i += offsetX)
            for (k = idy; k < nz; k += offsetY)
                edge_xz[i + k * nx] = data[_ind(i, j, k)];
    } else {
        if (edge_xz) {
            for (i = idx; i < nx; i += offsetX)
                for (k = idy; k < nz; k += offsetY)
                    data[_ind(i, j, k)] = edge_xz[i + k * nx];
        } else {
            for (i = idx; i < nx; i += offsetX)
                for (k = idy; k < nz; k += offsetY)
                    data[_ind(i, j, k)] = u;
        }
    }
}

__global__ void kernel_copy_edge_yz(double* edge_yz, double* data, int nx, int ny, int nz, int i, bool flag, double u) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;
    int j, k;
    if (flag) {
        for (k = idx; k < nz; k += offsetX)
            for (j = idy; j < ny; j += offsetY)
                edge_yz[j + k * ny] = data[_ind(i, j, k)];
    } else {
        if (edge_yz) {
            for (k = idx; k < nz; k += offsetX)
                for (j = idy; j < ny; j += offsetY)
                    data[_ind(i, j, k)] = edge_yz[j + k * ny];
        } else {
            for (k = idx; k < nz; k += offsetX)
                for (j = idy; j < ny; j += offsetY)
                    data[_ind(i, j, k)] = u;
        }
    }
}

__global__ void kernel_computation(double* next, double* data, int nx, int ny, int nz, double hx, double hy, double hz, double divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;
    int offsetZ = blockDim.z * gridDim.z;
    int i, j, k;

    for (i = idx; i < nx; i += offsetX) {
        for (j = idy; j < ny; j += offsetY) {
            for (k = idz; k < nz; k += offsetZ) {
                next[_ind(i, j, k)] = ((data[_ind(i - 1, j, k)] + data[_ind(i + 1, j, k)]) * hx + \
                                     (data[_ind(i, j - 1, k)] + data[_ind(i, j + 1, k)]) * hy + \
                                     (data[_ind(i, j, k - 1)] + data[_ind(i, j, k + 1)]) * hz) / \
                                     divisor;
            }
        }
    }
}

__global__ void kernel_error(double* next, double* data, double* diff, int nx, int ny, int nz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;
    int offsetZ = blockDim.z * gridDim.z;
    int i, j, k;

    for (i = idx - 1; i < nx + 1; i += offsetX) {
        for (j = idy - 1; j < ny + 1; j += offsetY) {
            for (k = idz - 1; k < nz + 1; k += offsetZ) {
                diff[_ind(i, j, k)] = (i != -1 && j != -1 && k != -1 && i != nx && j != ny && k != nz) * abs(next[_ind(i, j, k)] - data[_ind(i, j, k)]);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
    cout << fixed;
    cout.precision(7);

    // Input
    int i, j, k;
    double *data, *temp, *next;
    double *edge_xy, *edge_xz, *edge_yz;
    char proc_name[MPI_MAX_PROCESSOR_NAME];

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    int numproc, proc_name_len;
    // MPI initialisation
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Get_processor_name(proc_name, &proc_name_len);

    fprintf(stderr, "proc %2d(%d) on %s(%s)\n", id, numproc, proc_name, devProp.name);
    fflush(stderr);

    // int device_cnt;
    // cudaGetDeviceCount(&device_cnt);
    // cudaSetDevice(id % device_cnt);

    if (id == 0) {
        cin >> blocks[on_x] >> blocks[on_y] >> blocks[on_z];
        cin >> npx >> npy >> npz;
        cin >> filename;
        cin >> eps;
        cin >> l[on_x] >> l[on_y] >> l[on_z];
        cin >> u[down] >> u[up];
        cin >> u[left] >> u[right];
        cin >> u[front] >> u[back];
        cin >> u0;
    }

    // Передача параметров расчета всем процессам
    // MPI_Bcast(dimensions, NDIM, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&npx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&npy, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&npz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(blocks, NDIM, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(l, NDIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u, NDIM_2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // filename sending:
    int filename_size = filename.size();
    MPI_Bcast(&filename_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    filename.resize(filename_size);
    MPI_Bcast((char*) filename.c_str(), filename_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (blocks[on_x] * blocks[on_y] * blocks[on_z] * npx * npy * npz == 0) {
        fprintf(stderr, "Error at proc %d on %s\n", id, proc_name);
        if (blocks[on_x] * blocks[on_y] * blocks[on_z] != numproc) {
            fprintf(stderr, "Dead because of blocks\n");
            fprintf(stderr, "blocks[on_x]=%d, blocks[on_y]=%d, blocks[on_z]=%d, numproc=%d\n", blocks[on_x], blocks[on_y], blocks[on_z], numproc);
        }
        fflush(stderr);
        MPI_Finalize();
        return 0;
    }

    hx = l[on_x] / (double)(npx * blocks[on_x]);
    hy = l[on_y] / (double)(npy * blocks[on_y]);
    hz = l[on_z] / (double)(npz * blocks[on_z]);

    // We need hx^2 hy^2 hz^2
    double h2x = hx, h2y = hy, h2z = hz;
    h2x *= hx;
    h2y *= hy;
    h2z *= hz;

    // To a negative degree
    h2x = 1.0 / h2x;
    h2y = 1.0 / h2y;
    h2z = 1.0 / h2z;

    // Divisor as well
    double divisor = 2 * (h2x + h2y + h2z);
    // fprintf(stderr, "h2x=%f\n", h2x);
    // fprintf(stderr, "h2y=%f\n", h2y);
    // fprintf(stderr, "h2z=%f\n", h2z);
    // fprintf(stderr, "divisor=%f\n", divisor);
    // initiale bloks ids 3D
    ib = _ibx(id);
    jb = _iby(id);
    kb = _ibz(id);

    double* gpu_data, *gpu_next, *gpu_edge_xy, *gpu_edge_xz, *gpu_edge_yz;
    CUDA_ERROR(cudaMalloc(&gpu_data, sizeof(double) * (npx + 2) * (npy + 2) * (npz + 2)));
    CUDA_ERROR(cudaMalloc(&gpu_next, sizeof(double) * (npx + 2) * (npy + 2) * (npz + 2)));
    CUDA_ERROR(cudaMalloc(&gpu_edge_xy, sizeof(double) * npx * npy));
    CUDA_ERROR(cudaMalloc(&gpu_edge_xz, sizeof(double) * npx * npz));
    CUDA_ERROR(cudaMalloc(&gpu_edge_yz, sizeof(double) * npy * npz));


    // Buffer initialisation
    data = (double *)malloc(sizeof(double) * (npx + 2) * \
        (npy + 2) * (npz + 2));
    next = (double *)malloc(sizeof(double) * (npx + 2) * \
        (npy + 2) * (npz + 2));

    edge_xy = (double *)malloc(sizeof(double) * npx * npy);
    edge_xz = (double *)malloc(sizeof(double) * npx * npz);
    edge_yz = (double *)malloc(sizeof(double) * npy * npz);

    CUDA_ERROR(cudaMemcpy(gpu_edge_xy, edge_xy, sizeof(double) * npx * npy, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(gpu_edge_xz, edge_xz, sizeof(double) * npx * npz, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(gpu_edge_yz, edge_yz, sizeof(double) * npy * npz, cudaMemcpyHostToDevice));

    for (i = 0; i < npx; ++i) {
        for (j = 0; j < npy; ++j) {
            for  (k = 0; k < npz; ++k) {
                data[_i(i, j, k)] = u0;
                // fprintf(stderr, "%e ", data[_i(i, j, k)]);
            }
            // fprintf(stderr, "\n");
        }
    }
    // fflush(stderr);

    CUDA_ERROR(cudaMemcpy(gpu_data, data, sizeof(double) * (npx + 2) * (npy + 2) * (npz + 2), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(gpu_next, data, sizeof(double) * (npx + 2) * (npy + 2) * (npz + 2), cudaMemcpyHostToDevice));

    dim3 gblocks(32, 32);
    dim3 threads(32, 32);

    int dimensions[3];
    dimensions[0] = npx;
    dimensions[1] = npy;
    dimensions[2] = npz;
    CUDA_ERROR(cudaMemcpyToSymbol(g_dimensions, dimensions, 3 * sizeof(int)));

    double difference = 0.0;
    do {
        if (ib + 1 < blocks[on_x]) {
            kernel_copy_edge_yz<<<gblocks, threads>>>(gpu_edge_yz, gpu_data, npx, npy, npz, npx - 1, true, u0);
            CUDA_ERROR(cudaGetLastError());
            CUDA_ERROR(cudaMemcpy(edge_yz, gpu_edge_yz, sizeof(double) * npy * npz, cudaMemcpyDeviceToHost));
            MPI_Send(edge_yz, npy * npz, MPI_DOUBLE, _ib(ib + 1, jb, kb), 0, MPI_COMM_WORLD);
        }

        // Back
        if (jb + 1 < blocks[on_y]) {
            kernel_copy_edge_xz<<<gblocks, threads>>>(gpu_edge_xz, gpu_data, npx, npy, npz, npy - 1, true, u0);
            CUDA_ERROR(cudaGetLastError());
            CUDA_ERROR(cudaMemcpy(edge_xz, gpu_edge_xz, sizeof(double) * npx * npz, cudaMemcpyDeviceToHost));
            MPI_Send(edge_xz, npx * npz, MPI_DOUBLE, _ib(ib, jb + 1, kb), 0, MPI_COMM_WORLD);
        }

        // Up
        if (kb + 1 < blocks[on_z]) {
            kernel_copy_edge_xy<<<gblocks, threads>>>(gpu_edge_xy, gpu_data, npx, npy, npz, npz - 1, true, u0);
            CUDA_ERROR(cudaGetLastError());
            CUDA_ERROR(cudaMemcpy(edge_xy, gpu_edge_xy, sizeof(double) * npx * npy, cudaMemcpyDeviceToHost));
            MPI_Send(edge_xy, npx * npy, MPI_DOUBLE, _ib(ib, jb, kb + 1), 0, MPI_COMM_WORLD);
        }

        // Data recieve
        if (ib > 0) {
            MPI_Recv(edge_yz, npy * npz, MPI_DOUBLE, _ib(ib - 1, jb, kb), 0, MPI_COMM_WORLD, &status);
            CUDA_ERROR(cudaMemcpy(gpu_edge_yz, edge_yz, sizeof(double) * npy * npz, cudaMemcpyHostToDevice));
            kernel_copy_edge_yz<<<gblocks, threads>>>(gpu_edge_yz, gpu_data, npx, npy, npz, - 1, false, u0);
        } else {
            kernel_copy_edge_yz<<<gblocks, threads>>>(NULL, gpu_data, npx, npy, npz, - 1, false, u[left]);
        }
        CUDA_ERROR(cudaGetLastError());


        if (jb > 0) {
            MPI_Recv(edge_xz, npx * npz, MPI_DOUBLE, _ib(ib, jb - 1, kb), 0, MPI_COMM_WORLD, &status);
            CUDA_ERROR(cudaMemcpy(gpu_edge_xz, edge_xz, sizeof(double) * npx * npz, cudaMemcpyHostToDevice));
            kernel_copy_edge_xz<<<gblocks, threads>>>(gpu_edge_xz, gpu_data, npx, npy, npz, - 1, false, u0);
        } else {
            kernel_copy_edge_xz<<<gblocks, threads>>>(NULL, gpu_data, npx, npy, npz, - 1, false, u[front]);
        }
        CUDA_ERROR(cudaGetLastError());


        if (kb > 0) {
            MPI_Recv(edge_xy, npx * npy, MPI_DOUBLE, _ib(ib, jb, kb - 1), 0, MPI_COMM_WORLD, &status);
            CUDA_ERROR(cudaMemcpy(gpu_edge_xy, edge_xy, sizeof(double) * npx * npy, cudaMemcpyHostToDevice));
            kernel_copy_edge_xy<<<gblocks, threads>>>(gpu_edge_xy, gpu_data, npx, npy, npz, - 1, false, u0);
        } else {
            kernel_copy_edge_xy<<<gblocks, threads>>>(NULL, gpu_data, npx, npy, npz, - 1, false, u[down]);
        }
        CUDA_ERROR(cudaGetLastError());


        // Left
        if (ib > 0) {
            kernel_copy_edge_yz<<<gblocks, threads>>>(gpu_edge_yz, gpu_data, npx, npy, npz, 0, true, u0);
            CUDA_ERROR(cudaGetLastError());
            CUDA_ERROR(cudaMemcpy(edge_yz, gpu_edge_yz, sizeof(double) * npy * npz, cudaMemcpyDeviceToHost));
            MPI_Send(edge_yz, npy * npz, MPI_DOUBLE, _ib(ib - 1, jb, kb), 0, MPI_COMM_WORLD);
        }

        // Front
        if (jb > 0) {
            kernel_copy_edge_xz<<<gblocks, threads>>>(gpu_edge_xz, gpu_data, npx, npy, npz, 0, true, u0);
            CUDA_ERROR(cudaGetLastError());
            CUDA_ERROR(cudaMemcpy(edge_xz, gpu_edge_xz, sizeof(double) * npx * npz, cudaMemcpyDeviceToHost));
            MPI_Send(edge_xz, npx * npz, MPI_DOUBLE, _ib(ib, jb - 1, kb), 0, MPI_COMM_WORLD);
        }

        // Down
        if (kb > 0) {
            kernel_copy_edge_xy<<<gblocks, threads>>>(gpu_edge_xy, gpu_data, npx, npy, npz, 0, true, u0);
            CUDA_ERROR(cudaGetLastError());
            CUDA_ERROR(cudaMemcpy(edge_xy, gpu_edge_xy, sizeof(double) * npx * npy, cudaMemcpyDeviceToHost));
            MPI_Send(edge_xy, npx * npy, MPI_DOUBLE, _ib(ib, jb, kb - 1), 0, MPI_COMM_WORLD);
        }

        // Data recieve
        if (ib + 1 < blocks[on_x]) {
            MPI_Recv(edge_yz, npy * npz, MPI_DOUBLE, _ib(ib + 1, jb, kb), 0, MPI_COMM_WORLD, &status);
            CUDA_ERROR(cudaMemcpy(gpu_edge_yz, edge_yz, sizeof(double) * npy * npz, cudaMemcpyHostToDevice));
            kernel_copy_edge_yz<<<gblocks, threads>>>(gpu_edge_yz, gpu_data, npx, npy, npz, npx, false, u0);
        } else {
            kernel_copy_edge_yz<<<gblocks, threads>>>(NULL, gpu_data, npx, npy, npz, npx, false, u[right]);
        }
        CUDA_ERROR(cudaGetLastError());


        if (jb + 1 < blocks[on_y]) {
            MPI_Recv(edge_xz, npx * npz, MPI_DOUBLE, _ib(ib, jb + 1, kb), 0, MPI_COMM_WORLD, &status);
            CUDA_ERROR(cudaMemcpy(gpu_edge_xz, edge_xz, sizeof(double) * npx * npz, cudaMemcpyHostToDevice));
            kernel_copy_edge_xz<<<gblocks, threads>>>(gpu_edge_xz, gpu_data, npx, npy, npz, npy, false, u0);
        } else {
            kernel_copy_edge_xz<<<gblocks, threads>>>(NULL, gpu_data, npx, npy, npz, npy, false, u[back]);
        }
        CUDA_ERROR(cudaGetLastError());


        if (kb  + 1 < blocks[on_z]) {
            MPI_Recv(edge_xy, npx * npy, MPI_DOUBLE, _ib(ib, jb, kb + 1), 0, MPI_COMM_WORLD, &status);
            CUDA_ERROR(cudaMemcpy(gpu_edge_xy, edge_xy, sizeof(double) * npx * npy, cudaMemcpyHostToDevice));
            kernel_copy_edge_xy<<<gblocks, threads>>>(gpu_edge_xy, gpu_data, npx, npy, npz, npz, false, u0);
        } else {
            kernel_copy_edge_xy<<<gblocks, threads>>>(NULL, gpu_data, npx, npy, npz, npz, false, u[up]);
        }
        CUDA_ERROR(cudaGetLastError());

        cudaThreadSynchronize();
        // Recomputation
        kernel_computation<<<dim3(8, 8, 8), dim3(32, 4, 4)>>> (gpu_next, gpu_data, npx, npy, npz, h2x, h2y, h2z, divisor);
        CUDA_ERROR(cudaGetLastError());

        cudaThreadSynchronize();

        // Error
        double* gpu_difference;
        CUDA_ERROR(cudaMalloc((void**)&gpu_difference, sizeof(double) * (npx + 2) * (npy + 2) * (npz + 2)));
        kernel_error<<<dim3(8, 8, 8), dim3(32, 4, 4)>>> (gpu_next, gpu_data, gpu_difference, npx, npy, npz);
        CUDA_ERROR(cudaGetLastError());

        // fprintf(stderr, "Done gpu\n");
        // fflush(stderr);

        // Cast to thrust
        thrust::device_ptr< double > pointers = thrust::device_pointer_cast(gpu_difference);

        // Pointer of error
        thrust::device_ptr< double > res = thrust::max_element(pointers, pointers + (npx + 2) * (npy + 2) * (npz + 2));

        difference = 0.0;
        double gpu_diff = 0.0;

        // Get data from pointer
        gpu_diff = *res;

        temp = gpu_data;
        gpu_data = gpu_next;
        gpu_next = temp;

        MPI_Allreduce(&gpu_diff, &difference, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // fprintf(stderr, "difference=%f\n", difference);
        // fflush(stderr);

        CUDA_ERROR(cudaFree(gpu_difference));
    } while (difference > eps);

    fprintf(stderr, "Done computation\n");
    fflush(stderr);

    CUDA_ERROR(cudaMemcpy(data, gpu_data, sizeof(double) * (npx + 2) * (npy + 2) * (npz + 2), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaFree(gpu_data));
    CUDA_ERROR(cudaFree(gpu_next));
    CUDA_ERROR(cudaFree(gpu_edge_xy));
    CUDA_ERROR(cudaFree(gpu_edge_xz));
    CUDA_ERROR(cudaFree(gpu_edge_yz));

    // for (i = 0; i < dimensions[on_x]; ++i) {
    //     for (j = 0; j < dimensions[on_y]; ++j) {
    //         for  (k = 0; k < dimensions[on_z]; ++k) {
    //             // data[_i(i, j, k)] = u0;
    //             fprintf(stderr, "%e ", data[_i(i, j, k)]);
    //         }
    //         fprintf(stderr, "\n");
    //     }
    // }
    // fprintf(stderr, "\n");
    // fflush(stderr);

    int buff_size = (npx + 2) * (npy + 2) * (npz + 2);
    int new_symbol_size = 14;

    // Allocate mem
    char* buff = new char[buff_size * new_symbol_size];
    memset(buff, (char)' ', buff_size * new_symbol_size * sizeof(char));

    for (k = 0; k < dimensions[on_z]; ++k) {
        for (j = 0; j <  dimensions[on_y]; ++j) {
            int len_new_symbol;
            for (i = 0; i < dimensions[on_x] - 1; ++i) {
                len_new_symbol = sprintf(&buff[_i(i, j, k) * new_symbol_size], "%.6e", data[_i(i, j, k)]);
                // '\0' to ' ' (coz of new len_new_symbol)
                if (len_new_symbol < new_symbol_size) {
                    buff[_i(i, j, k) * new_symbol_size + len_new_symbol] = ' ';
                }
            }
            len_new_symbol = sprintf(&buff[_i(i, j, k) * new_symbol_size], "%.6e\n", data[_i(i, j, k)]);
            if(len_new_symbol < new_symbol_size){
                buff[_i(i, j, k) * new_symbol_size + len_new_symbol] = ' ';
            }
        }
    }
    /*
    for(i = 0; i < buff_size * new_symbol_size; ++i) {
        if (buff[i] == '\0') {

            buff[i] = ' ';
        }
        fprintf(stderr, "%с ", buff[i]);
    }
    */

    fprintf(stderr, "Done writting\n");
    fflush(stderr);

    MPI_Datatype new_representation;
    MPI_Datatype memtype;
    MPI_Datatype filetype;
    int sizes[NDIM], starts[NDIM], f_sizes[NDIM], f_starts[NDIM];

    MPI_Type_contiguous(new_symbol_size, MPI_CHAR, &new_representation);
    MPI_Type_commit(&new_representation);

    // Sizes for memtype
    sizes[on_x] = npx + 2;
    sizes[on_y] = npy + 2;
    sizes[on_z] = npz + 2;
    starts[on_x] = starts[on_y] = starts[on_z] = 1;

    // Sizes for filetype
    f_sizes[on_x] = dimensions[on_x] * blocks[on_x];
    f_sizes[on_y] = dimensions[on_y] * blocks[on_y];
    f_sizes[on_z] = dimensions[on_z] * blocks[on_z];

    f_starts[on_x] = dimensions[on_x] * ib;
    f_starts[on_y] = dimensions[on_y] * jb;
    f_starts[on_z] = dimensions[on_z] * kb;

    // Writting types
    // Memtype
    MPI_Type_create_subarray(3, sizes, dimensions, starts, MPI_ORDER_FORTRAN, new_representation, &memtype);
    MPI_Type_commit(&memtype);
    // Filetype
    MPI_Type_create_subarray(3, f_sizes, dimensions, f_starts, MPI_ORDER_FORTRAN, new_representation, &filetype);
    MPI_Type_commit(&filetype);

    fprintf(stderr, "Done creating\n");
    fflush(stderr);

    // Create and open file
    MPI_File fp;
    MPI_File_delete(filename.c_str(), MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fp);

    MPI_File_set_view(fp, 0, MPI_CHAR, filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fp, buff, 1, memtype, MPI_STATUS_IGNORE);

    MPI_File_close(&fp);

    fprintf(stderr, "Done writting in file\n");
    fflush(stderr);

    MPI_Finalize();

    if (id == 0) {
        fprintf(stderr, "%d %d %d\n", blocks[on_x], blocks[on_y], blocks[on_z]);
        fprintf(stderr, "%d %d %d\n", npx, npy, npz);
        fprintf(stderr, "%s\n", filename.c_str());
        fprintf(stderr, "%f\n", eps);
        fprintf(stderr, "%f %f %f\n", l[on_x], l[on_y], l[on_z]);
        fprintf(stderr, "%f %f\n", u[down], u[up]);
        fprintf(stderr, "%f %f\n", u[left], u[right]);
        fprintf(stderr, "%f %f\n", u[front], u[back]);
        fprintf(stderr, "%f\n", u0);
    }

    free(buff);
    free(data);
    free(next);
    free(edge_xy);
    free(edge_xz);
    free(edge_yz);

    return 0;
}
