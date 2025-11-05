# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport cos, sin, exp

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


cpdef double one_energy(cnp.ndarray[cnp.double_t, ndim=2] arr, int ix, int iy, int nmax): #typed memoryviews instead of NumPy arrays
    cdef double[:, :] a = arr
    return _one_energy(a, ix, iy, nmax)


cpdef double all_energy(cnp.ndarray[cnp.double_t, ndim=2] arr, int nmax): #operates directly on contiguous memory buffers, avoiding Python object creation
    cdef double[:, :] a = arr
    cdef int i, j
    cdef double enall = 0.0
    with nogil:
        for i in range(nmax):
            for j in range(nmax):
                enall += _one_energy(a, i, j, nmax)
    return enall


cpdef double get_order(cnp.ndarray[cnp.double_t, ndim=2] arr, int nmax): #Q-tensor components and trigonometric operations implemented using typed double variables
    cdef int i, j, a, b
    cdef cnp.ndarray[cnp.double_t, ndim=2] Qab = np.zeros((3, 3), dtype=np.float64)
    cdef double[:, :] Q = Qab
    cdef cnp.ndarray[cnp.double_t, ndim=3] lab = np.zeros((3, nmax, nmax), dtype=np.float64)
    cdef double[:, :, :] L = lab
    for i in range(nmax):
        for j in range(nmax):
            L[0, i, j] = cos(arr[i, j])
            L[1, i, j] = sin(arr[i, j])
            L[2, i, j] = 0.0
    cdef double s
    for a in range(3):
        for b in range(3):
            s = 0.0
            for i in range(nmax):
                for j in range(nmax):
                    s += 3.0 * L[a, i, j] * L[b, i, j] - (1.0 if a == b else 0.0)
            Q[a, b] = s / (2.0 * nmax * nmax)
    vals = np.linalg.eigvals(Qab).real
    return float(np.max(vals))


cpdef double MC_step(cnp.ndarray[cnp.double_t, ndim=2] arr, double Ts, int nmax): #core Metropolis step rewritten in Cython to reduce overhead from Python loops and random generation
    cdef double[:, :] a = arr                                                     #uses C API for random numbers and array access through C memoryviews
    cdef double scale = 0.1 + Ts
    cdef int i, j, ix, iy, accept = 0
    cdef double ang, en0, en1, boltz, u
    xran = np.random.randint(0, nmax, size=(nmax, nmax), dtype=np.int32)
    yran = np.random.randint(0, nmax, size=(nmax, nmax), dtype=np.int32)
    aran = np.random.normal(0.0, scale, size=(nmax, nmax))
    uran = np.random.random((nmax, nmax))
    cdef int[:, :] mx = xran
    cdef int[:, :] my = yran
    cdef double[:, :] ma = aran
    cdef double[:, :] mu = uran
    for i in range(nmax):
        for j in range(nmax):
            ix = mx[i, j]
            iy = my[i, j]
            ang = ma[i, j]
            en0 = _one_energy(a, ix, iy, nmax)
            a[ix, iy] += ang
            en1 = _one_energy(a, ix, iy, nmax)
            if en1 <= en0:
                accept += 1
            else:
                boltz = exp(-(en1 - en0) / Ts)
                u = mu[i, j]
                if boltz >= u:
                    accept += 1
                else:
                    a[ix, iy] -= ang
    return accept / (nmax * nmax)


cpdef cnp.ndarray initdat(int nmax):
    return np.random.random_sample((nmax, nmax)) * (2.0 * np.pi)


def plotdat(cnp.ndarray[cnp.double_t, ndim=2] arr, int pflag, int nmax):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    if pflag == 0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
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
    ax.set_aspect('equal')
    plt.show()


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
    for i in range(nsteps + 1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i, float(ratio[i]), float(energy[i]), float(order[i])), file=f)
    f.close()
    return filename


cpdef main(int iterations, int nmax, double Ts, int pflag):
    import time
    lattice = initdat(nmax).astype(np.float64, copy=False)
    plotdat(lattice, pflag, nmax)
    energy = np.zeros(iterations + 1, dtype=np.double)
    ratio = np.zeros(iterations + 1, dtype=np.double)
    order = np.zeros(iterations + 1, dtype=np.double)
    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice, nmax)
    t0 = time.time()
    for it in range(1, iterations + 1):
        ratio[it] = MC_step(lattice, Ts, nmax)
        energy[it] = all_energy(lattice, nmax)
        order[it] = get_order(lattice, nmax)
    runtime = time.time() - t0
    print(f"Cython.py: Size: {nmax}, Steps: {iterations}, T*: {Ts:.3f}: Order: {order[iterations-1]:.3f}, Time: {runtime:.6f} s")
    savedat(lattice, iterations, Ts, runtime, ratio, energy, order, nmax)
    plotdat(lattice, pflag, nmax)
