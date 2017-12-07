# https://www.datacamp.com/community/tutorials/machine-learning-python
# Machine Learning tutorial for basic data manipulation etc
# Collection of series of commands

# Import `datasets` from `sklearn`
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load in the `digits` data
digits = datasets.load_digits()

# Load in the data with read_csv()
# If data is dled using pandas, it is already split up into the training
# and testing sets, `.tra` and `.tes` https://www.datacamp.com/courses/importing-data-in-python-part-1/
training_data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header = None)

def gather_basic_info_data():

    # you can visually check that the images and the data are
    # related by reshaping the images array to 2 dims
    digits.images.reshape((1797, 64))

    # can also just check if its equal
    print(np.all(digits.images.reshape((1797,64))==digits.data))

    digits_keys = digits.keys()
    digits_data = digits.data
    digits_target = digits.target
    digits_DESCR = digits.DESCR

    # n samples, m features
    print(digits_data.shape)

    # n samples => n target values
    print(digits_target.shape)

    number_digits = len(np.unique(digits.target))
    digits_images = digits.images

    # n unique values => 0 to n-1. i.e all target values are made up
    # of nums between 0 to n-1
    print(number_digits)

    # n instances, x pixels by y pixels big
    print(digits_images.shape)

    # dictionary keys of the data
    print(digits.keys())

    # data
    print(digits.data)

    # target values
    print(digits.target)

    # description of the digits data
    print(digits.DESCR)

    # check the type of the data
    print(type(digits.data))


def visualise_matplotlib():
    # issue with display variable????

    # Figure size (width, height) in inches
    fig = plt.figure(figsize=(6,6))

    # Adjust the subplots
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # For each of the 64 images
    for i in range(64):

        # Initialise the subplots: add a subplot in the grid of 8 by 8, at the
        # i + 1-th position
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])

        # Display an image at the i-th position
        ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

        # label the image with the target values
        ax.text(0, 7, str(digits.target[i]))

    # Show the plot
    plt.savefig("picture.jpg")


if __name__ == "__main__":
    visualise_matplotlib()
    quit()
    gather_basic_info_data()
