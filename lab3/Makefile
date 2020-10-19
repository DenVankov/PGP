CC = nvcc
CFLAGS = -Werror cross-execution-space-call -lm -rdc=true
FILES = test.cu
PROG = lab3

all:
	$(CC) $(CFLAGS) -o $(PROG) $(FILES)
