
def stats(table):


    # default axis is "index" -> axis = 0
    # we can change the axis to "column" -> axis = 1

    # Calculates the mean along each column
    print(table.mean())

    # Calculates the mean along each row
    print(table.mean(axis=1))

    print(table.quantile(0.75))

    newtable = table[["AUS", "CAN", "JPN"]]

    # Returns the general basic statistics of each column
    # that was requested
    print(newtable.describe())

    print(table.min())       # find min of each column

    # returns the row index labels which has the minimum of each column
    print(table.idxmin())

    
