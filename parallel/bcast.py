from mpi4py import MPI

def main():

    #MPI.COMM_WORLD -> intracommunicator
    #intracommunicator is within a block of threads/processes
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        #broadcast arg1 to all the other nodes from node specified by
        #arg2
        msg = comm.bcast("Hello from Rank 0", root=0)
    else:

        #the string in arg1 doesn't affect the output...
        #play around with root values to get differents bcast responses
        msg = comm.bcast("what?", root=3)

    print("Rank {} received: {}".format(rank, msg))

if __name__ == "__main__":
    main()
