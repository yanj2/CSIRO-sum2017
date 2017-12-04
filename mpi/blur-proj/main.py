import numpy as np
from scipy import misc
from scipy import ndimage as img
from skimage import io
from mpi4py import MPI
import sys

boxfilter = 1/9 * np.ones((3,3))
edge2 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
edge1= np.array([[0,1,0],[1,-4,1],[0,1,0]])
identity = np.array([[0,0,0],[0,1,0],[0,0,0]])
sobelx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
sobely = sobelx.transpose()

# main function that controls the flow of the program
def main():

    # retrieves the commandline arguments
    args = sys.argv

    # retrieves the expected imgsrc argument
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

    # calculate the horizontal and vertial gradients of the image
    edgex = imgfilter(src, sobelx)
    edgey = imgfilter(src, sobely)

    # calculate the gradient magnitudes of the image
    result = np.sqrt(edgex ** 2 + edgey ** 2)

    # normalise the adjusted pixel values
    result = (result - np.min(result))/(np.max(result)-np.min(result)) * 255

    # save the final image in a new jpg file
    skimage.io.imsave("output.jpg", result.astype(np.uint8))

# helper function that applies the given "kern" filter to the provided src
# numpy array
def imgfilter(src, kern, version=0):

    dim = src.shape        # retrieve the dimensions of the image
    copy = src.copy()      # copy of src image that holds new values

    # find the offset value needed based on the kernel dimensions
    # assuming that the kernel is square
    offset = np.floor(kern.shape[0]/2).astype(np.int)

    # using nested for loops to modify the pixel values
    if (version == 0):

        for col in range(offset, dim[0]-offset):
            for row in range(offset, dim[1]-offset):
                copy[col, row] = np.sum(src[col-offset:col+offset+1
                           ,row-offset:row+offset+1] * kern, axis=(1,0))

    # using loop and joblib to parallelise the process of editing the pixels
    elif (version == 1):

        for col in range(offset, dim[0]-offset):
            Parallel



    # return final edited image array
    return copy

def

# always keep main at the bottom so that all the functions have been read prior
# to running the content inside main
if __name__ == "__main__":
    main()
