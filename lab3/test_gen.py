from random import *

if __name__ == '__main__':
    # low test
    name = 'low'
    f = open('tests/' + name + '.t', 'w')
    f.write(f"img/150x150.data\n")
    f.write(f"img/150x150_out.data\n")
    n = 30
    sz = 150 * 150
    s = set()
    f.write(f"{n}\n")
    for _ in range(n):
        a = 500
        f.write(f"{a} ")
        for _ in range(a):
            while True:
                x = randint(0, 149)
                y = randint(0, 149)
                if (x, y) not in s:
                    break
            s.add((x, y))
            f.write(f"{x} {y} ")
        f.write("\n")
    f.write("\n")
    f.close()

    # mid test
    name = 'mid'
    f = open('tests/' + name + '.t', 'w')
    f.write(f"img/600x600.data\n")
    f.write(f"img/600x600_out.data\n")
    n = 30
    sz = 600 * 600
    s = set()
    f.write(f"{n}\n")
    for _ in range(n):
        a = 10000
        f.write(f"{a} ")
        for _ in range(a):
            while True:
                x = randint(0, 600 - 1)
                y = randint(0, 600 - 1)
                if (x, y) not in s:
                    break
            s.add((x, y))
            f.write(f"{x} {y} ")
        f.write("\n")
    f.write("\n")
    f.close()


    # high test
    name = 'high'
    f = open('tests/' + name + '.t', 'w')
    f.write(f"img/3840x2400.data\n")
    f.write(f"img/3840x2400_out.data\n")
    n = 30
    sz = 3840 * 2400
    s = set()
    f.write(f"{n}\n")
    for _ in range(n):
        a = 200000
        f.write(f"{a} ")
        for _ in range(a):
            while True:
                x = randint(0, 3840 - 1)
                y = randint(0, 2400 - 1)
                if (x, y) not in s:
                    break
            s.add((x, y))
            f.write(f"{x} {y} ")
        f.write("\n")
    f.write("\n")
    f.close()
