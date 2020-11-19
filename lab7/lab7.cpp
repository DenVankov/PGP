#include "mpi.h"
#include <bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>

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

// Индексация внутри блока
#define _i(i, j, k) (((k) + 1) * (dimensions[on_y] + 2) * (dimensions[on_x] + 2) + ((j) + 1) * (dimensions[on_x] + 2) + (i) + 1)
#define _iz(id) (((id) / (dimensions[on_x] + 2) / (dimensions[on_y] + 2)) - 1)
#define _iy(id) ((((id) % ((dimensions[on_x] + 2) * (dimensions[on_y] + 2))) / (dimensions[on_x] + 2)) - 1)
#define _ix(id) ((id) % (dimensions[on_x] + 2) - 1)

// Индексация по блокам (процессам)
#define _ib(i, j, k) ((k) * blocks[on_y] * blocks[on_x] + (j) * blocks[on_x] + (i))
#define _ibz(id) ((id) / blocks[on_x] / blocks[on_y])
#define _iby(id) (((id) % (blocks[on_x] * blocks[on_y])) / blocks[on_x])
#define _ibx(id) ((id) % blocks[on_x])

const int root = 0;
const int NDIM = 3;
const int NDIM_2 = 6;

int main(int argc, char *argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
    cout << fixed;
    cout.precision(7);

    int id, ib, jb, kb;
    int i, j, k;

    // Input
    int dimensions[NDIM];
    int blocks[NDIM];
    double l[NDIM];
    double u[NDIM_2];
    double eps, u0;
    string filename;

    double hx, hy, hz;
    double *data, *temp, *next, *buff;
    MPI_Status status;

    int numproc;
    // MPI initialisation
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if (id == 0) {
        cin >> blocks[on_x] >> blocks[on_y] >> blocks[on_z];
        cin >> dimensions[on_x] >> dimensions[on_y] >> dimensions[on_z];
        cin >> filename;
        cin >> eps;
        cin >> l[on_x] >> l[on_y] >> l[on_z];
        cin >> u[down] >> u[up];
        cin >> u[left] >> u[right];
        cin >> u[front] >> u[back];
        cin >> u0;
    }

    // Передача параметров расчета всем процессам
    MPI_Bcast(dimensions, NDIM, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(blocks, NDIM, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(l, NDIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u, NDIM_2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    hx = l[on_x] / (double)(dimensions[on_x] * blocks[on_x]);
    hy = l[on_y] / (double)(dimensions[on_y] * blocks[on_y]);
    hz = l[on_z] / (double)(dimensions[on_z] * blocks[on_z]);

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

    // initiale bloks ids 3D
    ib = _ibx(id);
    jb = _iby(id);
    kb = _ibz(id);
    // fprintf(stderr, "\nWORKING ON ID=%d WITH ib=%d, jb=%d, kb=%d\n", id, ib, jb, kb);
    // fflush(stderr);

    // Buffer initialisation
    data = (double *)malloc(sizeof(double) * (dimensions[on_x] + 2) * \
        (dimensions[on_y] + 2) * (dimensions[on_z] + 2));
    next = (double *)malloc(sizeof(double) * (dimensions[on_x] + 2) * \
        (dimensions[on_y] + 2) * (dimensions[on_z] + 2));

    long long mx = max(max(dimensions[on_x], dimensions[on_y]), dimensions[on_z]);
    buff = (double *)malloc(sizeof(double) * (mx + 2) * (mx + 2));

    for (i = 0; i < dimensions[on_x]; ++i) {
        for (j = 0; j < dimensions[on_y]; ++j) {
            for  (k = 0; k < dimensions[on_z]; ++k) {
                data[_i(i, j, k)] = u0;
                // fprintf(stderr, "\ndata[_i(i, j, k)]=%f\nu0=%f\n", data[_i(i, j, k)], u0);
                // fflush(stderr);
            }
        }
    }

    // int it = 0;
    double difference = 0.0;
    do {
        // Data send
        // Right
        if (ib + 1 < blocks[on_x]) {
            for (k = 0; k < dimensions[on_z]; ++k) {
                for (j = 0; j < dimensions[on_y]; ++j) {
                    buff[j + k * dimensions[on_y]] = data[_i(dimensions[on_x] - 1, j, k)];
                }
            }
            MPI_Send(buff, dimensions[on_y] * dimensions[on_z], MPI_DOUBLE,
                _ib(ib + 1, jb, kb), 0, MPI_COMM_WORLD);
        }

        // Up
        if (jb + 1 < blocks[on_y]) {
            for (k = 0; k < dimensions[on_z]; ++k)
                for (i = 0; i < dimensions[on_x]; ++i)
                    buff[i + k * dimensions[on_x]] = data[_i(i, dimensions[on_y] - 1, k)];
            MPI_Send(buff, dimensions[on_x] * dimensions[on_z], MPI_DOUBLE,
                _ib(ib, jb + 1, kb), 0, MPI_COMM_WORLD);
        }

        // Back
        if (kb + 1 < blocks[on_z]) {
            for (j = 0; j < dimensions[on_y]; ++j)
                for (i = 0; i < dimensions[on_x]; ++i)
                    buff[i + j * dimensions[on_x]] = data[_i(i, j, dimensions[on_z] - 1)];
            MPI_Send(buff, dimensions[on_x] * dimensions[on_y], MPI_DOUBLE,
                _ib(ib, jb, kb + 1), 0, MPI_COMM_WORLD);
        }


	    if (ib > 0) {
            MPI_Recv(buff, dimensions[on_y] * dimensions[on_z], MPI_DOUBLE,
            _ib(ib - 1, jb, kb), 0, MPI_COMM_WORLD, &status);
            for (k = 0; k < dimensions[on_z]; ++k)
                for (j = 0; j < dimensions[on_y]; ++j)
                    data[_i(-1, j, k)] = buff[j + k * dimensions[on_y]];
        } else {
            for (k = 0; k < dimensions[on_z]; ++k)
                for (j = 0; j < dimensions[on_y]; ++j)
                    data[_i(-1, j, k)] = u[left];
        }

        if (jb > 0) {
            MPI_Recv(buff, dimensions[on_x] * dimensions[on_z], MPI_DOUBLE,
                _ib(ib, jb - 1, kb), 0, MPI_COMM_WORLD, &status);
            for (k = 0; k < dimensions[on_z]; ++k)
                for (i = 0; i < dimensions[on_x]; ++i)
                    data[_i(i, -1, k)] = buff[i + k * dimensions[on_x]];
        } else {
            for (k = 0; k < dimensions[on_z]; ++k)
                for (i = 0; i < dimensions[on_x]; ++i)
                    data[_i(i, -1, k)] = u[front];
        }

        if (kb > 0) {
            MPI_Recv(buff, dimensions[on_x] * dimensions[on_y], MPI_DOUBLE,
                _ib(ib, jb, kb - 1), 0, MPI_COMM_WORLD, &status);
            for (j = 0; j < dimensions[on_y]; ++j)
                for (i = 0; i < dimensions[on_x]; ++i)
                    data[_i(i, j, -1)] = buff[i + j * dimensions[on_x]];
        } else {
            for (j = 0; j < dimensions[on_y]; ++j)
                for (i = 0; i < dimensions[on_x]; ++i)
                    data[_i(i, j, -1)] = u[down];
        }

        // Left
        if (ib > 0) {
            for (k = 0; k < dimensions[on_z]; ++k)
                for (j = 0; j < dimensions[on_y]; ++j)
                    buff[j + k * dimensions[on_y]] = data[_i(0, j, k)];
            MPI_Send(buff, dimensions[on_y] * dimensions[on_z], MPI_DOUBLE,
                _ib(ib - 1, jb, kb), 0, MPI_COMM_WORLD);
        }

        // Down
        if (jb > 0) {
            for (k = 0; k < dimensions[on_z]; ++k)
                for (i = 0; i < dimensions[on_x]; ++i)
                    buff[i + k * dimensions[on_x]] = data[_i(i, 0, k)];
            MPI_Send(buff, dimensions[on_x] * dimensions[on_z], MPI_DOUBLE,
                _ib(ib, jb - 1, kb), 0, MPI_COMM_WORLD);
        }

        // Front
        if (kb > 0) {
            for (j = 0; j < dimensions[on_y]; ++j)
                for (i = 0; i < dimensions[on_x]; ++i)
                    buff[i + j * dimensions[on_x]] = data[_i(i, j, 0)];
            MPI_Send(buff, dimensions[on_x] * dimensions[on_y], MPI_DOUBLE,
                _ib(ib, jb, kb - 1), 0, MPI_COMM_WORLD);
        }

        // Data recieve


        if (ib + 1 < blocks[on_x]) {
            MPI_Recv(buff, dimensions[on_y] * dimensions[on_z], MPI_DOUBLE,
                _ib(ib + 1, jb, kb), 0, MPI_COMM_WORLD, &status);
            for (k = 0; k < dimensions[on_z]; ++k)
                for (j = 0; j < dimensions[on_y]; ++j)
                    data[_i(dimensions[on_x], j, k)] = buff[j + k * dimensions[on_y]];
        } else {
            for (k = 0; k < dimensions[on_z]; ++k)
                for (j = 0; j < dimensions[on_y]; ++j)
                    data[_i(dimensions[on_x], j, k)] = u[right];
        }

        if (jb + 1 < blocks[on_y]) {
            MPI_Recv(buff, dimensions[on_x] * dimensions[on_z], MPI_DOUBLE,
                _ib(ib, jb + 1, kb), 0, MPI_COMM_WORLD, &status);
            for (k = 0; k < dimensions[on_z]; ++k)
                for (i = 0; i < dimensions[on_x]; ++i)
                    data[_i(i, dimensions[on_y], k)] = buff[i + k * dimensions[on_x]];
        } else {
            for (k = 0; k < dimensions[on_z]; ++k)
                for (i = 0; i < dimensions[on_x]; ++i)
                    data[_i(i, dimensions[on_y], k)] = u[back];
        }

        if (kb  + 1 < blocks[on_z]) {
            MPI_Recv(buff, dimensions[on_x] * dimensions[on_y], MPI_DOUBLE,
                _ib(ib, jb, kb + 1), 0, MPI_COMM_WORLD, &status);
            for (j = 0; j < dimensions[on_y]; ++j)
                for (i = 0; i < dimensions[on_x]; ++i)
                    data[_i(i, j, dimensions[on_z])] = buff[i + j * dimensions[on_x]];
        } else {
            for (j = 0; j < dimensions[on_y]; ++j)
                for (i = 0; i < dimensions[on_x]; ++i)
                    data[_i(i, j, dimensions[on_z])] = u[up];
        }

        // Recomputation
        difference = 0.0;
        for (i = 0; i < dimensions[on_x]; ++i) {
            for (j = 0; j < dimensions[on_y]; ++j) {
                for (k = 0; k < dimensions[on_z]; ++k) {
                    next[_i(i, j, k)] = ((data[_i(i - 1, j, k)] + data[_i(i + 1, j, k)]) * h2x + \
                                         (data[_i(i, j - 1, k)] + data[_i(i, j + 1, k)]) * h2y + \
                                         (data[_i(i, j, k - 1)] + data[_i(i, j, k + 1)]) * h2z) / \
                                         divisor;
                    difference = max(difference, abs(next[_i(i, j, k)] - data[_i(i, j, k)]));
                    // fprintf(stderr, "\ndiff=%f", difference);
                    // fflush(stderr);
                }
            }
        }

        // double tmp = 0.0;
        MPI_Allreduce(&difference, &difference, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        // difference = tmp;

        temp = next;
        next = data;
        data = temp;

        // if (id == root) {
        //     fprintf(stderr, "it = %d, difference = %.10f\n", it, difference);
        //     fflush(stderr);
        // }
        // it++;
        // if (it > 500)
        //     break;
    } while (difference > eps);

    if (id != 0) {
        for (k = 0; k < dimensions[on_z]; ++k) {
            for(j = 0; j < dimensions[on_y]; ++j) {
                for(i = 0; i < dimensions[on_x]; ++i) {
                    buff[i] = data[_i(i, j, k)];
                }
                MPI_Send(buff, dimensions[on_x], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        // freopen(filename.c_str(), "w", stdout);
        FILE* file;
        file = fopen(filename.c_str(), "w");
        // fprintf(stderr, "\nFILENAME: %s\n", filename.c_str());
        // fflush(stderr);
        for (kb = 0; kb < blocks[on_z]; ++kb) {
            for (k = 0; k < dimensions[on_z]; ++k) {
                for (jb = 0; jb < blocks[on_y]; ++jb) {
                    for(j = 0; j < dimensions[on_y]; ++j) {
                        for (ib = 0; ib < blocks[on_x]; ++ib) {

                            // Check if we are on 0 pr recieve data from all
                            if (_ib(ib, jb, kb) == 0) {
                                for (i = 0; i < dimensions[on_x]; ++i) {
                                    buff[i] = data[_i(i, j, k)];
                                }
                            } else {
                                MPI_Recv(buff, dimensions[on_x], MPI_DOUBLE,
                                     _ib(ib, jb, kb), 0, MPI_COMM_WORLD, &status);
                            }

                            // Printing data
                            for (i = 0; i < dimensions[on_x]; ++i) {
                                // cout << buff[i] << " ";
                                fprintf(file, "%.7e ", buff[i]);
                            }
                        }
                    }
                }
            }
        }
        fclose(file);
    }

    MPI_Finalize();

    free(buff);
    free(data);
    free(next);

    return 0;
}
