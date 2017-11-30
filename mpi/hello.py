from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print ("hello world from process {}/{}".format(rank, size))

if __name__ == "__main__":
    main()
