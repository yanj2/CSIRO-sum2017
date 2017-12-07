import dataframe
import pivot
import np
import fn
import descr_stats
from pandas import read_csv

def main():
    # Read tabular data from a file.
    data = read_csv("oecd_stats.csv")

    table = pivot.create_pivot_table(data)

    quit()
    descr_stats.stats(table)
    np.extract_np(table)

    dataframe.manipulate_data(data)
    dataframe.create_dataframe()
    dataframe.access_data(data)
    dataframe.extract_data(data)


if __name__ == "__main__":
    main()
