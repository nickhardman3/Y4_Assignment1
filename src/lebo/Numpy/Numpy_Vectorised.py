import os
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def one_energy(arr, ix, iy, nmax):
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax

    e = 0.0
    for nx, ny in ((ixp, iy), (ixm, iy), (ix, iyp), (ix, iym)):
        d = arr[ix, iy] - arr[nx, ny]
        e += -0.5 * (3.0 * np.cos(d)**2 - 1.0)
    return e

def all_energy(arr, nmax): #simplified to a fully vectorised global energy computation using np.roll for neighbour alignment
    up    = np.roll(arr, -1, axis=0)
    down  = np.roll(arr,  1, axis=0)
    left  = np.roll(arr, -1, axis=1)
    right = np.roll(arr,  1, axis=1)

    s = (3.0 * np.cos(arr - up)**2   - 1.0)
    s += (3.0 * np.cos(arr - down)**2 - 1.0)
    s += (3.0 * np.cos(arr - left)**2 - 1.0)
    s += (3.0 * np.cos(arr - right)**2 - 1.0)

    return -0.25 * np.sum(s)

def get_order(arr, nmax): #Q-tensor and order parameter computed directly using NumPy mean and broadcasting
    cx = np.cos(arr)
    sy = np.sin(arr)

    Qxx = 0.5 * (3.0 * np.mean(cx*cx) - 1.0)
    Qyy = 0.5 * (3.0 * np.mean(sy*sy) - 1.0)
    Qxy = 1.5 * np.mean(cx*sy)

    Q = np.array([[Qxx, Qxy],
                  [Qxy, Qyy]])
    vals = np.linalg.eigvalsh(Q)
    return float(vals[-1])

def MC_step(arr, Ts, nmax): #fully vectorised Monte Carlo update using array masks for checkerboard updates
    scale = 0.1 + Ts
    accept_total = 0
    for parity in (0, 1): #replaces explicit loops with NumPy broadcasting and np.roll for neighbour calculations
        mask = (np.add.outer(np.arange(nmax), np.arange(nmax)) % 2 == parity)

        proposal = arr.copy()
        proposal[mask] += np.random.normal(loc=0.0, scale=scale, size=mask.sum())

        up, down = np.roll(arr, -1, 0), np.roll(arr, 1, 0)
        left, right = np.roll(arr, -1, 1), np.roll(arr, 1, 1)
        E0 = -0.5 * ((3*np.cos(arr - up)**2 - 1) +
                     (3*np.cos(arr - down)**2 - 1) +
                     (3*np.cos(arr - left)**2 - 1) +
                     (3*np.cos(arr - right)**2 - 1))

        up_n, down_n = np.roll(proposal, -1, 0), np.roll(proposal, 1, 0)
        left_n, right_n = np.roll(proposal, -1, 1), np.roll(proposal, 1, 1)
        E1 = -0.5 * ((3*np.cos(proposal - up_n)**2 - 1) +
                     (3*np.cos(proposal - down_n)**2 - 1) +
                     (3*np.cos(proposal - left_n)**2 - 1) +
                     (3*np.cos(proposal - right_n)**2 - 1))

        dE = E1 - E0

        rand = np.random.random(arr.shape)
        accept_mask = mask & ((dE <= 0) | (rand < np.exp(-dE / Ts)))
        arr[accept_mask] = proposal[accept_mask]
        accept_total += np.count_nonzero(accept_mask)

    return accept_total / (2 * nmax * nmax)

def initdat(nmax):
    return np.random.random_sample((nmax, nmax)) * 2.0 * np.pi

def plotdat(arr, pflag, nmax): #modified to use array-based operations for computing local energies and colour maps
    if pflag == 0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)

    if pflag == 1: #uses np.roll for neighbouring sites rather than recomputation in loops
        mpl.rc('image', cmap='rainbow')
        up    = np.roll(arr, -1, axis=0)
        down  = np.roll(arr,  1, axis=0)
        left  = np.roll(arr, -1, axis=1)
        right = np.roll(arr,  1, axis=1)
        cols = -0.5 * (
            (3.0 * np.cos(arr - up)**2    - 1.0) +
            (3.0 * np.cos(arr - down)**2  - 1.0) +
            (3.0 * np.cos(arr - left)**2  - 1.0) +
            (3.0 * np.cos(arr - right)**2 - 1.0)
        )
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

    energy = np.zeros(nsteps + 1, dtype=np.float64)
    ratio  = np.zeros(nsteps + 1, dtype=np.float64)
    order  = np.zeros(nsteps + 1, dtype=np.float64)

    energy[0] = all_energy(lattice, nmax)
    ratio[0]  = 0.5
    order[0]  = get_order(lattice, nmax)

    t0 = time.time()
    for it in range(1, nsteps + 1):
        ratio[it]  = MC_step(lattice, temp, nmax)
        energy[it] = all_energy(lattice, nmax)
        order[it]  = get_order(lattice, nmax)
    runtime = time.time() - t0

    print(f"{program}: Size: {nmax:d}, Steps: {nsteps:d}, T*: {temp:5.3f}: "
          f"Order: {order[-1]:5.3f}, Time: {runtime:8.6f} s")

    savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)
    plotdat(lattice, pflag, nmax)

if __name__ == "__main__":
    if len(sys.argv) == 5:
        PROG = sys.argv[0]
        STEPS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMP = float(sys.argv[3])
        PFLAG = int(sys.argv[4])
        main(PROG, STEPS, SIZE, TEMP, PFLAG)
    else:
        print(f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")
