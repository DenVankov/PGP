CC = nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm -rdc=true -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__
FILES = gaus.cu
PROG = lab4

all:
	$(CC) $(CFLAGS) -o $(PROG) $(FILES)
