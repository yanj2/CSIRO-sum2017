def extract_np(table):
    print(table.index)    # creates Row Index object
    print(type(table.values))

    # you can use normal integer indexing for numpy arrays
    print(table.values[:3,:3])

    # you can use standard array indexing on the rows
    print(table[:3])

    # indexing using the row label
    print(table.ix[2006])

    # you can even select multiple columns!!
    print(table.ix[:,["AUS","SWE"]])

    # you can select multiple rows and columns. INCLUDES 2007 and COL!!!!!
    print(table.ix[2000:2007, "AUS":"COL"])

    # unsuuurreee... need to double check
    start, end = table.index[[1,5]]
    table.ix[start:end, ["AUS"]]
    print(table.head())

    # returns all the rows in the column AUS
    # SERIES from dataframe
    print(table.ix[:, "AUS"])

    # DATAFRAME with a single column from a data frame
    print(table.ix[:, ["AUS"]])

    # boolean indexing
    # here specifically returns all the rows where AUS has a values
    # greater than 500000
    subset = table[table.AUS > 500000]
    print(subset.head())
