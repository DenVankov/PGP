import random

if __name__ == '__main__':
    lst = [(0, 10), (10, 15), (15, 16)]
    data = [(100, 1000), (1000000, 8000000), (500000000, 100000000)]
    idx = 0
    for pack in lst:
        for test in range(pack[0], pack[1]):
            n = random.randint(data[idx][0], data[idx][1])
            name = str(test)
            file = open('tests/' + name + '.t', 'w')
            file.write(f"{n}\n")
            if (idx != 2):
                for _ in range(0, n):
                    file.write(f"{random.uniform(-100000.0, 100000.0)} ")
            else:
                for _ in range(0, n):
                    file.write(f"{random.uniform(-1000.0, 1000.0)} ")
            file.write("\n")
            file.close()
        idx += 1
