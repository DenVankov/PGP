#include <bits/stdc++.h>
#include <chrono>

using namespace std;

__global__ void kernel(float *arr, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;			// Абсолютный номер потока
	int offset = blockDim.x * gridDim.x;						// Общее кол-во потоков
	for(int i = idx; i < n; i += offset) {
		if (arr[i] < 0)
            arr[i] = abs(arr[i]);
    }
}

int main() {
	vector<int> v {32, 128, 512, 1024};
	for (int idx = 0; idx < v.size(); ++idx) {
		for (int j = 0; j < 16; ++j) {
			string str = to_string(j);
			string name = "tests/" + str + ".t";
			freopen(name.c_str(), "r", stdin);
			freopen("log_gpu.txt", "a", stdout);
			long long n;
	    	scanf("%lld", &n);
			cout << "GPU:\n";
			cout << "n: " << n << "\n";
			cout << "Threads: " << v[idx] << ", " << v[idx] << "\n";
			float *arr = (float *)malloc(sizeof(float) * n);
			for(long long i = 0; i < n; ++i) {
				scanf("%f", &arr[i]);
	    	}

			float *dev_arr;
			cudaMalloc(&dev_arr, sizeof(float) * n);
			cudaMemcpy(dev_arr, arr, sizeof(float) * n, cudaMemcpyHostToDevice);

			auto start = chrono::steady_clock::now();
			kernel<<<v[idx], v[idx]>>>(dev_arr, n);
			auto end = chrono::steady_clock::now();

			cudaMemcpy(arr, dev_arr, sizeof(float) * n, cudaMemcpyDeviceToHost);
			cudaFree(dev_arr);
			free(arr);
			cout << ((double)chrono::duration_cast<chrono::microseconds>(end - start).count()) / 1000.0 << "ms\n\n";
		}
	}
	return 0;
}
