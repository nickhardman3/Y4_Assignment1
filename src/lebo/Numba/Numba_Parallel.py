import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit, prange

@njit(nopython=True, cache=True)
def one_energy(arr, ix, iy, nmax):
    en = 0.0
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax
    ang = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang)**2)
    ang = arr[ix, iy] - arr[ixm, iy]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang)**2)
    ang = arr[ix, iy] - arr[ix, iyp]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang)**2)
    ang = arr[ix, iy] - arr[ix, iym]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang)**2)
    return en

@njit(nopython=True, parallel=True, cache=True) #parallel=True to distribute outer loop (i) across threads using prange()
def all_energy(arr, nmax):
    enall = 0.0
    for i in prange(nmax):
        subtotal = 0.0
        for j in range(nmax):
            subtotal += one_energy(arr, i, j, nmax)
        enall += subtotal
    return enall

@njit(nopython=True, parallel=True, cache=True)
def get_order(arr, nmax): #parallelised over lattice rows using prange() for both lab vector construction and tensor summation
    Qab = np.zeros((3, 3))
    delta = np.eye(3)
    lab = np.empty((3, nmax, nmax))

    for i in prange(nmax):
        for j in range(nmax):
            lab[0, i, j] = np.cos(arr[i, j])
            lab[1, i, j] = np.sin(arr[i, j])
            lab[2, i, j] = 0.0

    for a in range(3):
        for b in range(3):
            s = 0.0
            for i in prange(nmax):
                subtotal = 0.0
                for j in range(nmax):
                    subtotal += 3 * lab[a, i, j] * lab[b, i, j] - delta[a, b]
                s += subtotal
            Qab[a, b] = s
    Qab = Qab / (2 * nmax * nmax)
    vals, _ = np.linalg.eig(Qab)
    smax = np.max(np.real(vals))
    return smax

@njit(nopython=True, cache=True)
def MC_step(arr, Ts, nmax):
    scale = 0.1 + Ts
    accept = 0
    xran = np.random.randint(0, nmax, (nmax, nmax))
    yran = np.random.randint(0, nmax, (nmax, nmax))
    aran = np.random.normal(0.0, scale, (nmax, nmax))

    for i in range(nmax):
        for j in range(nmax):
            ix = xran[i, j]
            iy = yran[i, j]
            ang = aran[i, j]
            en0 = one_energy(arr, ix, iy, nmax)
            old = arr[ix, iy]
            arr[ix, iy] = old + ang
            en1 = one_energy(arr, ix, iy, nmax)
            if en1 <= en0:
                accept += 1
            else:
                boltz = np.exp(-(en1 - en0) / Ts)
                if boltz >= np.random.random():
                    accept += 1
                else:
                    arr[ix, iy] = old
    return accept / (nmax * nmax)

def initdat(nmax):
    return np.random.random_sample((nmax, nmax)) * 2.0 * np.pi

def plotdat(arr, pflag, nmax):
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

def savedat(arr, nsteps, Ts, runtime, ratio, energy, order, nmax):
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = f"LL-Output-{current_datetime}.txt"
    with open(filename, "w") as f:
        print("#=====================================================", file=f)
        print(f"# File created:        {current_datetime}", file=f)
        print(f"# Size of lattice:     {nmax}x{nmax}", file=f)
        print(f"# Number of MC steps:  {nsteps}", file=f)
        print(f"# Reduced temperature: {Ts:5.3f}", file=f)
        print(f"# Run time (s):        {runtime:8.6f}", file=f)
        print("#=====================================================", file=f)
        print("# MC step:  Ratio:     Energy:   Order:", file=f)
        print("#=====================================================", file=f)
        for i in range(nsteps + 1):
            print(f"   {i:05d}    {ratio[i]:6.4f} {energy[i]:12.4f}  {order[i]:6.4f} ", file=f)

def main(program, nsteps, nmax, temp, pflag):
    lattice = initdat(nmax)
    plotdat(lattice, pflag, nmax)
    energy = np.zeros(nsteps + 1)
    ratio = np.zeros(nsteps + 1)
    order = np.zeros(nsteps + 1)
    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice, nmax)
    start = time.time()
    for it in range(1, nsteps + 1):
        ratio[it] = MC_step(lattice, temp, nmax)
        energy[it] = all_energy(lattice, nmax)
        order[it] = get_order(lattice, nmax)
    runtime = time.time() - start
    print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp:5.3f}: Order: {order[-1]:5.3f}, Time: {runtime:8.6f} s")
    savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)
    plotdat(lattice, pflag, nmax)

if __name__ == '__main__':
    if len(sys.argv) == 5:
        main(sys.argv[0], int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
    else:
        print(f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")
