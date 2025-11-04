import sys
from MPI_Cython import main

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"Usage: mpirun -n <ranks> python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")
        sys.exit(1)
    iterations = int(sys.argv[1])
    nmax = int(sys.argv[2])
    Ts = float(sys.argv[3])
    pflag = int(sys.argv[4])
    main(iterations, nmax, Ts, pflag)
