import numpy as np
from pandas import Series, DataFrame

def create_dataframe():

    # Create a data frame using a dictionary
    df = DataFrame({"A":range(3), "B":range(3)}, index=("X", "Y", "Z"))
    print(df)

    # Create a data frame by combining Series
    series1 = Series(range(3), index=("X", "Y", "Z"))
    series2 = Series(range(3), index=("X", "Y", "Z"))
    df = DataFrame({"A":series1, "B":series2})
    print(df)

def access_data(data):

    print(data.head())          # Return the first 5 rows
    print(data.tail())          # Return the last 5 rows

    # We can specify the number of rows to Return
    print(data.head(10))

    print(data.columns)         # Returns column index object
    print(data.columns[1])      # Can index the column index object
    print(data.columns[:-3])    # Can slice the column index object

    # Treats the data frame like a dictionary with the Column heading as
    # the key and the rest of the column as values.
    # `unique()` ignores any repeated values
    print(data["LOCATION"].unique())

    # A column is a Series object.
    # Note that this can be tested with `type` as usual

    # Returns a column as an object attribute of a data frame.
    print(data.LOCATION.unique())

def manipulate_data(data):

    # Inserting new columns at the end of the data DataFrame
    data["Random"] = data["Value"]/200
    print(data.head())

    # You can file the entire column with a single value using:
    data["Random"] = 1

    # Delete a columns
    del data["SUBJECT"]
    del data["Flag Codes"]
    print(data.head())

def extract_data(data):

    # Boolean indexing to extract rows matching certain criteria
    # Essentially gives a data frame that tells you which rows match the
    # criteria
    index_arr = data['TIME'] == 1960
    print(index_arr)

    # Returns all the rows in the table that are True in the Boolean table
    print(data[index_arr])
