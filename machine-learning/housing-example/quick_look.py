import pandas as pd
import get_data
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import os

def methods():

    # NOTE:if you exclue the (), then you're just pointing the variable at the
    # function. Nothing is happening otherwise
    data = get_data.load_housing_data()

    # shows with districts belong in this category and how many per district
    print(data["ocean_proximity"].value_counts())

    # describe summarises some basic statistics
    # NOTE: NULL values are ignored in count
    print(data.describe())

    # matplotlib - visualise with histogram
    data.hist(bins=50, figsize=(20,15))

    PATH = os.getcwd()
    if not os.path.isdir("graphs"):
        os.makedirs("graphs")

    plt.savefig(os.path.join(PATH + "/graphs/", "histogram.png"))


if __name__ == "__main__":
    methods()
