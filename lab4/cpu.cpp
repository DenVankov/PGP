#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

int main(int argc, char const *argv[]) {
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
            cout << data[i + j * n] << " ";
        }
        cout << "\n";
    }

    cout << "\n";

    for (auto &j: p)
        cout << j << " ";

    cout << "\n";
    return 0;
}
