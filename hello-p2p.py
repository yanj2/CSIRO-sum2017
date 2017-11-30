from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        for i in range(1,size):
            sendMsg = "Hello, Rank {}".format(i)
            comm.send(sendMsg, dest=i)
    else: 
        recvMsg = comm.recv(source=0)
        print(recvMsg)

if __name__ == "__main__":
    main()

