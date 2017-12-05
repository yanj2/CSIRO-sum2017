import numpy as np
from scipy import ndimage as img
from skimage import io
import sys
import time
from joblib import Parallel, delayed

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

boxfilter = 1/9 * np.ones((3,3))
edge2 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
edge1= np.array([[0,1,0],[1,-4,1],[0,1,0]])
identity = np.array([[0,0,0],[0,1,0],[0,0,0]])
sobelx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
sobely = sobelx.transpose()

SERIAL = 0
JOBLIB1 = 1
JOBLIB2 = 2

clock = Timer()

# main function that controls the flow of the program
def main():

    # retrieves the commandline arguments
    args = sys.argv

    # retrieves the expected img srcfile argument
    if (len(args) >= 3):
        print ("Invalid number of inputs")
        quit()

    try:
        src = img.imread(args[1], flatten=True)

    except:
        print ("Failed to load {}".format(args[1]))
        quit()

    if src is None:
        print ("Failed to load {}".format(args[1]))
        quit()

    for n in range(1,src.shape[0]):

        # calculate the horizontal and vertial gradients of the image
        edgex = imgfilter(src, sobelx, n, JOBLIB2)
        edgey = imgfilter(src, sobely, n, JOBLIB2)

    """
    # for test runs of new implementations
    edgex = imgfilter(src, sobelx, 4, JOBLIB2)
    edgey = imgfilter(src, sobely, 4, JOBLIB2)
    """
    
    # calculate the gradient magnitudes of the image
    result = np.sqrt(edgex ** 2 + edgey ** 2)

    # normalise the adjusted pixel values
    result = (result - np.min(result))/(np.max(result)-np.min(result)) * 255

    # save the final image in a new jpg file
    io.imsave("output.jpg", result.astype(np.uint8))

    # write the computation times for different numbers of workers into a file
    runstats("joblib-version2")

# write the computation times for different numbers of workers into a file
def runstats(name):

    f = open("stats.txt", "a")
    f.write("\n----{}----\n".format(name))

    for n in range(np.floor(len(clock.times)/2).astype(int)):
        f.write("{0:5}  :  {1:3}\n".format(n+1,clock.times[n] + clock.times[n+1]))

    f.write("-"*(len(name)+8) + "\n")
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

    # using loop and joblib to parallelise the process of editing the pixels
    elif (version == JOBLIB1):

        clock.starttime()

        # creates n workers that are reused in all calls of parallel
        with Parallel(n_jobs=workers) as parallel:
            for col in range(offset, dim[0]-offset):

                # !! find a way to increase the work completed by the function
                copy[col,offset:dim[1]-offset] = np.array(parallel(delayed(editcell)(src, col, row, offset, kern)
                                                 for row in range(offset, dim[1]-offset)))

        clock.addtime()
        clock.times.append(clock.gettime())
        clock.reset()

    elif (version == JOBLIB2):

        clock.starttime()

        with Parallel(n_jobs=workers) as parallel:

            copy = np.array(parallel(delayed(edit)(src, offset, kern, col)
                   for col in range(offset, dim[0]-offset)))

        clock.addtime()
        clock.times.append(clock.gettime())
        clock.reset()

    # return final edited image array
    return copy

# calculate the new values for each column
def edit(src, offset, kern, col):

    temp = src.copy()[col] # temp storage of the column being calc

    for row in range(offset, src.shape[1]-offset):
        temp[row] = np.sum(src[col-offset:col+offset+1,
                        row-offset:row+offset+1] * kern, axis=(1,0))

    return temp

# calculate the new value of the cell based on the given kern
def editcell(src, col, row, offset, kern):
    return np.sum(src[col-offset:col+offset+1, row-offset:row+offset+1] * kern, axis=(1,0))


# always keep main at the bottom so that all the functions have been declared
# prior to running the content inside main
if __name__ == "__main__":
    main()
