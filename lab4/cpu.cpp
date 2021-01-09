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
        double mx = DBL_MIN;
        int idx = i;
        for (int j = i; j < n; ++j) {
            double uii = data[j + i * n];
            for (int q = 0; q < i; ++q) {
                uii -= data[j + q * n] * data[q + j * n];
            }
            if (abs(uii) > mx) {
                mx = abs(uii);
                idx = j;
                fprintf(stderr, "MAX=%f\n", mx);
            }
        }

        p[i] = idx;
        for (int j = 0; j < n; ++j) {
                swap(data[i + j * n], data[idx + j * n]);
        }

        // Got U
        for (int j = i; j < n; ++j) {
            for (int q = 0; q < i; ++q) {
                data[i + j * n] -= data[i + q * n] * data[q + j * n];
            }
        }

        // Got L
        for (int j = i + 1; j < n; ++j) {
            for (int q = 0; q < i; ++ q) {
                data[j + i * n] -= data[j + q * n] * data[q + i * n];
            }
            data[j + i * n] /= data[i + i * n];
        }

        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                // cout << data[i + j * n] << " ";
                printf("%.10e ", data[k + j * n]);
            }
            printf("\n");
        }
        printf("\n");
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // cout << data[i + j * n] << " ";
            printf("%.10e ", data[i + j * n]);
        }
        cout << "\n";
    }

    cout << "\n";

    for (auto &j: p)
        cout << j << " ";

    cout << "\n";
    return 0;
}
