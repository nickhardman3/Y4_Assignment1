# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport cos, sin, exp
from cython.parallel cimport prange #openMP-based parallelisation within Cython
cimport openmp

cnp.import_array()

cdef inline double _one_energy(double[:, :] arr, int ix, int iy, int nmax) nogil:
    cdef int ixp = (ix + 1) % nmax
    cdef int ixm = (ix - 1 + nmax) % nmax
    cdef int iyp = (iy + 1) % nmax
    cdef int iym = (iy - 1 + nmax) % nmax
    cdef double en = 0.0
    cdef double ang, c
    ang = arr[ix, iy] - arr[ixp, iy]; c = cos(ang); en += 0.5 * (1.0 - 3.0 * c * c)
    ang = arr[ix, iy] - arr[ixm, iy]; c = cos(ang); en += 0.5 * (1.0 - 3.0 * c * c)
    ang = arr[ix, iy] - arr[ix, iyp]; c = cos(ang); en += 0.5 * (1.0 - 3.0 * c * c)
    ang = arr[ix, iy] - arr[ix, iym]; c = cos(ang); en += 0.5 * (1.0 - 3.0 * c * c)
    return en


cpdef double one_energy(cnp.ndarray[cnp.double_t, ndim=2] arr, int ix, int iy, int nmax):
    cdef double[:, :] a = arr
    return _one_energy(a, ix, iy, nmax)


cpdef double all_energy(cnp.ndarray[cnp.double_t, ndim=2] arr, int nmax):
    cdef double[:, :] a = arr
    cdef Py_ssize_t idx, nn = nmax * nmax
    cdef int i, j
    cdef double total = 0.0, s
    with nogil:
        s = 0.0
        for idx in prange(nn, schedule='static'): #parallelise outer loop over rows
            i = <int>(idx // nmax)
            j = <int>(idx - i * nmax)
            s += _one_energy(a, i, j, nmax)
        total = s
    return total


cpdef double get_order(cnp.ndarray[cnp.double_t, ndim=2] arr, int nmax): #compiled with OpenMP support for parallelisable trigonometric operations
    cdef int i, j
    cdef Py_ssize_t idx, nn = nmax * nmax
    cdef cnp.ndarray[cnp.double_t, ndim=2] Qab = np.zeros((3, 3), dtype=np.float64)
    cdef double[:, :] Q = Qab
    cdef cnp.ndarray[cnp.double_t, ndim=2] cx = np.empty((nmax, nmax), dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=2] sx = np.empty((nmax, nmax), dtype=np.float64)
    cdef double[:, :] CX = cx
    cdef double[:, :] SX = sx
    cdef double[:, :] A = arr

    with nogil:
        for idx in prange(nn, schedule='static'):
            i = <int>(idx // nmax)
            j = <int>(idx - i * nmax)
            CX[i, j] = cos(A[i, j])
            SX[i, j] = sin(A[i, j])

    cdef double s
    with nogil:
        s = 0.0
        for idx in prange(nn, schedule='static'):
            i = <int>(idx // nmax); j = <int>(idx - i * nmax)
            s += 3.0 * CX[i, j] * CX[i, j] - 1.0
        Q[0, 0] = s / (2.0 * nmax * nmax)

        s = 0.0
        for idx in prange(nn, schedule='static'):
            i = <int>(idx // nmax); j = <int>(idx - i * nmax)
            s += 3.0 * CX[i, j] * SX[i, j]
        Q[0, 1] = s / (2.0 * nmax * nmax)

        s = 0.0
        for idx in prange(nn, schedule='static'):
            i = <int>(idx // nmax); j = <int>(idx - i * nmax)
            s += 3.0 * SX[i, j] * CX[i, j]
        Q[1, 0] = s / (2.0 * nmax * nmax)

        s = 0.0
        for idx in prange(nn, schedule='static'):
            i = <int>(idx // nmax); j = <int>(idx - i * nmax)
            s += 3.0 * SX[i, j] * SX[i, j] - 1.0
        Q[1, 1] = s / (2.0 * nmax * nmax)

        Q[0, 2] = 0.0
        Q[1, 2] = 0.0
        Q[2, 0] = 0.0
        Q[2, 1] = 0.0
        Q[2, 2] = -0.5 

    cdef cnp.ndarray[cnp.double_t, ndim=1] vals = np.linalg.eigvals(Qab).real
    return float(np.max(vals))


cpdef double MC_step(cnp.ndarray[cnp.double_t, ndim=2] arr, double Ts, int nmax): #random updates and acceptance checks handled at C-level for reduced overhead
    cdef double[:, :] a = arr
    cdef double scale = 0.1 + Ts
    cdef int parity, i, j, start
    cdef double en0, en1, ang, u, boltz
    cdef double accept_total = 0.0
    cdef double acc_local, acc_row

    angles = np.random.normal(0.0, scale, size=(nmax, nmax))
    uniforms = np.random.random((nmax, nmax))
    cdef double[:, :] MA = angles
    cdef double[:, :] MU = uniforms

    with nogil:
        for parity in range(2):  
            acc_local = 0.0
            for i in prange(nmax, schedule='static'):
                start = (i + parity) & 1
                acc_row = 0.0
                for j in range(start, nmax, 2):
                    ang = MA[i, j]
                    u = MU[i, j]
                    en0 = _one_energy(a, i, j, nmax)
                    a[i, j] += ang
                    en1 = _one_energy(a, i, j, nmax)

                    if en1 <= en0:
                        acc_row += 1.0
                    else:
                        boltz = exp(-(en1 - en0) / Ts)
                        if boltz >= u:
                            acc_row += 1.0
                        else:
                            a[i, j] -= ang
                acc_local += acc_row
            accept_total += acc_local

    return accept_total / (nmax * nmax)


cpdef cnp.ndarray initdat(int nmax):
    return np.random.random_sample((nmax, nmax)) * (2.0 * np.pi)


def plotdat(cnp.ndarray[cnp.double_t, ndim=2] arr, int pflag, int nmax):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    if pflag == 0:
        return
    u = np.cos(arr); v = np.sin(arr)
    x = np.arange(nmax); y = np.arange(nmax)
    cols = np.zeros((nmax, nmax))
    if pflag == 1:
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i, j] = one_energy(arr, i, j, nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag == 2:
        mpl.rc('image', cmap='hsv')
        cols = arr % np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)
    quiveropts = dict(headlength=0, pivot='middle', headwidth=1, scale=1.1 * nmax)
    fig, ax = plt.subplots()
    ax.quiver(x, y, u, v, cols, norm=norm, **quiveropts)
    ax.set_aspect('equal'); plt.show()


cpdef savedat(cnp.ndarray[cnp.double_t, ndim=2] arr, int nsteps, double Ts, double runtime, cnp.ndarray ratio, cnp.ndarray energy, cnp.ndarray order, int nmax):
    import datetime
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    f = open(filename, "w")
    print("#=====================================================", file=f)
    print("# File created:        {:s}".format(current_datetime), file=f)
    print("# Size of lattice:     {:d}x{:d}".format(nmax, nmax), file=f)
    print("# Number of MC steps:  {:d}".format(nsteps), file=f)
    print("# Reduced temperature: {:5.3f}".format(Ts), file=f)
    print("# Run time (s):        {:8.6f}".format(runtime), file=f)
    print("#=====================================================", file=f)
    print("# MC step:  Ratio:     Energy:   Order:", file=f)
    print("#=====================================================", file=f)
    cdef int i
    for i in range(nsteps + 1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(
            i, float(ratio[i]), float(energy[i]), float(order[i])), file=f)
    f.close()
    return filename


cpdef main(int iterations, int nmax, double Ts, int pflag, int threads):
    import time
    if threads > 0:
        openmp.omp_set_num_threads(threads)

    lattice = initdat(nmax).astype(np.float64, copy=False)
    plotdat(lattice, pflag, nmax)
    energy = np.zeros(iterations + 1, dtype=np.double)
    ratio = np.zeros(iterations + 1, dtype=np.double)
    order = np.zeros(iterations + 1, dtype=np.double)

    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice, nmax)

    t0 = time.time()
    cdef int it
    for it in range(1, iterations + 1):
        ratio[it] = MC_step(lattice, Ts, nmax)
        energy[it] = all_energy(lattice, nmax)
        order[it] = get_order(lattice, nmax)
    runtime = time.time() - t0

    print(f"Cython_omp.py: Size: {nmax}, Steps: {iterations}, T*: {Ts:.3f}: "
          f"Order: {order[iterations-1]:.3f}, Time: {runtime:.6f} s")

    savedat(lattice, iterations, Ts, runtime, ratio, energy, order, nmax)
    plotdat(lattice, pflag, nmax)
