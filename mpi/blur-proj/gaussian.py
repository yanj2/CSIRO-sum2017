import math

# function that returns % of pixel values
def blurFactor(n):

    # currently hardcoded one dimensional gaussian kernel based on
    # size 1 sigma(std) and kernel size 5
    # http://dev.theomader.com/gaussian-kernel-calculator/
    oneDKernel = [0.061, 0.245, 0.388, 0.245, 0.061]

    return oneDKernel[n]
