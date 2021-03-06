import numpy as np
from scipy import ndimage as img
from skimage import io
import sys
import time
from joblib import Parallel, delayed
from mpi4py import MPI
import os
import platform
"""
Class Timer that handles keeping track of how long it takes for functions to
run. Used specifically for comparing the difference in speed between serial
and parallel versions of the code
"""
class Timer:

    times = []

    def __init__(self):
        self.start = time.clock()
        self.end = time.clock()

    def starttime(self):
        self.start = time.clock()

    def addtime(self):
        self.end += time.clock() - self.end

    def gettime(self):
        return self.end - self.start

    def reset(self):
        self.start = time.clock()
        self.end = time.clock()

    def fullreset(self):
        self.times = []

sobelx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
sobely = sobelx.transpose()

SERIAL = 0
JOBLIB1 = 1
JOBLIB2 = 2
NESTED = 3
MPIPY = 4
clock = Timer()

# main function that controls the flow of the program
def main(imgfile):

    src = img.imread(imgfile[1]+'/'+imgfile[0], flatten=True)

    edgex = imgfilter(src, sobelx, 6, JOBLIB2)
    edgey = imgfilter(src, sobely, 6, JOBLIB2)

    # calculate the gradient magnitudes of the image
    result = np.sqrt(edgex ** 2 + edgey ** 2)

    # normalise the adjusted pixel values
    result = (result - np.min(result))/(np.max(result)-np.min(result)) * 255

    # create a filename to save file
    filename = "output-files/" + imgfile[0]

    # save the final image in a new jpg file
    io.imsave(filename, result.astype(np.uint8))


# manages the parallel processes, iterates over each image file in the chosen
# dir, delegates an image file to a process dependent on their rank and index
# of the image file
def handler():

    path = os.getcwd()+"/src-files"    # locates the directory for src files
    srcdir = os.listdir(path)          # creates a list of files in the dir

    comm = MPI.COMM_WORLD              # creates a communicator for the process
    rank = comm.Get_rank()             # variable for the rank of curr process
    size = comm.Get_size()             # variable for total number of processes

    pos = rank                         # init position for process in srcdir

    # run the imgfilter function on all images delegated to this process
    while pos < len(srcdir):
        print("i am rank {}, on node {}".format(rank, platform.node()))
        main((srcdir[pos], path))
        #runstats(srcdir[pos][:-4])
        pos += size

def handler_send():
    # some issues that need to be resolved with this version of MPI

    path = os.getcwd()+"/src-files"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        srcdir = os.listdir(path)
        for idx in range(len(srcdir)):
            comm.isend(((srcdir[idx], path), len(srcdir)-(idx+1)), dest=idx % size + 1)

    else:

        data = comm.recv(source=0)
        while data[1]:
            print("I am rank {}".format(rank))
            main(data[0])

# write the computation times for different numbers of workers into a file
def runstats(name):
    # need to update this code to cater for multiple images

    f = open("stats/parallel-test-stats.txt", "a")
    f.write("\n----{}----\n".format(name))

    for n in range(np.floor(len(clock.times)/2).astype(int)):
        f.write("{0:5}  :  {1:3.8}\n".format(n+1,clock.times[n] + clock.times[n+1]))

    f.write("-"*(len(name)+8) + "\n")
    clock.fullreset()
    f.close()

# helper function that applies the given "kern" filter to the provided src
# numpy array
def imgfilter(src, kern, workers, version=SERIAL):

    dim = src.shape        # retrieve the dimensions of the image
    copy = src.copy()      # copy of src image that holds new values

    # find the offset value needed based on the kernel dimensions
    # assuming that the kernel is square
    offset = np.floor(kern.shape[0]/2).astype(np.int)

    # using nested for loops to modify the pixel values
    if (version == SERIAL):

        clock.starttime()

        for col in range(offset, dim[0]-offset):
            for row in range(offset, dim[1]-offset):
                copy[col, row] = np.sum(src[col-offset:col+offset+1,
                                 row-offset:row+offset+1] * kern, axis=(1,0))

        clock.addtime()
        clock.times.append(clock.gettime())
        clock.reset()

    # creates workers to calculate the new values of each columns
    elif (version == JOBLIB2):

        clock.starttime()

        # creates n workers that are reused in all calls of the function
        with Parallel(n_jobs=workers) as parallel:

            copy = np.array(parallel(delayed(edit)(src, offset, kern, col)
                   for col in range(offset, dim[0]-offset)))

        clock.addtime()
        clock.times.append(clock.gettime())
        clock.reset()

    return copy

#------------------------FUNCTIONS CALLS FOR PARALLEL-------------------------
# calculate the new values for each column
def edit(src, offset, kern, col):

    temp = src.copy()[col] # temp storage of the column being calc

    # calculate the new value of each row element
    for row in range(offset, src.shape[1]-offset):
        temp[row] = np.sum(src[col-offset:col+offset+1,
                        row-offset:row+offset+1] * kern, axis=(1,0))

    # returns an array of all the new column values
    return temp
#-----------------------------------------------------------------------------
# always keep main at the bottom so that all the functions have been declared
# prior to running the content inside main
if __name__ == "__main__":
    handler()
