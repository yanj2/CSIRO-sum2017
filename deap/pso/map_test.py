from scoop import futures

def fun(a,b):
    return a,b

if __name__ == "__main__":
    a = [1,2,3,4]
    b = [1,2,3,4]
    print(list(futures.map(fun,a,b)))
