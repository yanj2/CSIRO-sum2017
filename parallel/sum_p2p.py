from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    val = (rank+1)*10
    print ("Rank {} has value {}".format(rank,val))

    if rank == 0:

        #retrieves the data sent by other processes
        sum0 = val
        for i in range(1,size):
            sum0 += comm.recv(source=i)

        print("Rank 0 worked out the total {}".format(sum0))

    else:

        #sends information to the destination of 0...
        comm.send(val, dest=0)


if __name__ == "__main__":
    main()
