#include <bits/stdc++.h>
#include <chrono>

using namespace std;

int main() {
	for (int j = 0; j < 16; ++j) {
		string str = to_string(j);
		string name = "tests/" + str + ".t";
		freopen(name.c_str(), "r", stdin);
		freopen("log.txt", "a", stdout);
		uint32_t n;
	    scanf("%d", &n);
		fprintf(stderr, "%d\n", n);
		cout << "CPU:\n";
		cout << "n: " << n << "\n";
		auto start = chrono::steady_clock::now();
		float *arr = (float *)malloc(sizeof(float) * n);
		for (uint32_t i = 0; i < n; ++i) {
			scanf("%f", &arr[i]);
			arr[i] = abs(arr[i]);
	    }
		auto end = chrono::steady_clock::now();

		free(arr);
		cout << ((double)chrono::duration_cast<chrono::microseconds>(end - start).count()) / 1000.0 << "ms\n\n";
	}
	return 0;
}
