import numpy as np
from pandas import Series, DataFrame, pivot_table

def create_pivot_table(data):
    # https://stackoverflow.com/questions/47152691/how-to-pivot-a-dataframe
    # help manage issues with pivot tables
    if data.duplicated(["LOCATION", "TIME"]).any():
        table = pivot_table(data, index="TIME", columns="LOCATION", values="Value")
    else:
        table = data.pivot(index="LOCATION", columns="TIME", values="Value")
    #print(table.head())

    # Forcing printing a wide table... but looks messy
    #print(table.head(3).to_string())

    return table
