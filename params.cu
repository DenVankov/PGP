#include <bits/stdc++.h>

int main() {
    int Dev;
    cudaGetDeviceCount(&Dev);
    for (int i = 0; i < Dev; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i);
      printf("\tDevice name: %s\n", prop.name);
      printf("\tTotalGlobalMem: %lu\n", prop.totalGlobalMem);
      printf("\tConst Mem : %lu\n", prop.totalConstMem);
      printf("Max shared mem for blocks %lu\n", prop.sharedMemPerBlock);
      printf("Max regs per block %d\n", prop.regsPerBlock);
      printf("Max thread per block %d\n", prop.maxThreadsPerBlock);
      printf("MultiProcessorCount : %d\n", prop.multiProcessorCount);
      printf("MaxThreadsDim %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
      printf("MaxGridSize %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}
