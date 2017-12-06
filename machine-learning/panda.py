import numpy as np
from pandas import Series, DataFrame, read_csv

# Create a data frame using a dictionary
df = DataFrame({"A":range(3), "B":range(3)}, index=("X", "Y", "Z"))
print(df)

# Create a data frame by combining Series
series1 = Series(range(3), index=("X", "Y", "Z"))
series2 = Series(range(3), index=("X", "Y", "Z"))
df = DataFrame({"A":series1, "B":series2})
print(df)

# Read tabular data from a file.
data = read_csv("oecd_stats.csv")
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
