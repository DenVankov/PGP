import matplotlib.pyplot as plt

cpu = ([], [])
gpu = {}
PATH_CPU = "log.txt"
PATH_GPU = "log_gpu.txt"

def draw_gpu():
    fig, ax = plt.subplots()
    for key in gpu.keys():
        tuple = gpu[key]
        plt.plot(tuple[0], tuple[1], label=key)
    plt.title("GPU threads compare")
    plt.savefig('gpu.png')
    plt.legend(loc='best')
    plt.xlabel("Size, n")
    plt.ylabel("Time, ms")
    plt.grid()
    plt.show()

def draw_cpu():
    fig, ax = plt.subplots()
    for key in gpu.keys():
        tuple = gpu[key]
        plt.plot(tuple[0], tuple[1], label=key)
    plt.plot(cpu[0], cpu[1], label="CPU")
    plt.title("GPU with CPU compare")
    plt.savefig('cpu.png')
    plt.legend(loc='best')
    plt.xlabel("Size, n")
    plt.ylabel("Time, ms")
    plt.grid()
    plt.show()


with open(PATH_GPU, 'r') as f:
    while (True):
        check = f.readline()
        if not check:
            break
        line = f.readline()
        n = int(line.split(" ")[1])
        line = (f.readline()).split(" ")
        thread = int(line[2])
        line = (f.readline()).split("m")
        ms = float(line[0])
        f.readline()
        if thread not in gpu:
            gpu[thread] = ([], [])
        gpu[thread][0].append(n)
        gpu[thread][1].append(ms)

with open(PATH_CPU, 'r') as f:
    while (True):
        check = f.readline()
        if not check:
            break
        line = f.readline()
        n = int(line.split(" ")[1])
        line = (f.readline()).split("m")
        ms = float(line[0])
        f.readline()
        cpu[0].append(n)
        cpu[1].append(ms)

draw_gpu()
draw_cpu()
