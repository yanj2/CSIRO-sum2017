import sys
import os

def main():

    path = os.getcwd()+"/slurm-out"
    srcdir = os.listdir(path)

    for filename in srcdir:
        stream = read_file(path + "/" + filename)
        extract(stream)

def extract(stream):
    time_lst = stream[-4:-1]
    user_time = time_lst[1].split()[1][:-1].replace("m", " ")
    sys_time = time_lst[2].split()[1][:-1].replace("m", " ")

    time = (float(user_time.split()[0]) + float(sys_time.split()[0]),
            float(user_time.split()[1]) + float(sys_time.split()[1]))

    f = open("time.txt","a")
    f.write("\n{0}\n{1:2}m{2:.3f}s\n".format(str(stream[-1]),time[0], time[1]))
    f.close()

def read_file(filename):

    try:
        fp = open(filename)
    except PermissionError:
        return "wot"
    else:
        with fp:
            return fp.readlines()


if __name__ == "__main__":
    main()
