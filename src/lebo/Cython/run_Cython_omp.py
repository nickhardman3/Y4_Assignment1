import sys
from Cython_omp import main

if len(sys.argv) != 6:
    print(f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADS>")
    sys.exit(1)

iterations = int(sys.argv[1])
nmax = int(sys.argv[2])
Ts = float(sys.argv[3])
pflag = int(sys.argv[4])
threads = int(sys.argv[5])

main(iterations, nmax, Ts, pflag, threads)