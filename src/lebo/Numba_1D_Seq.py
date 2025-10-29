import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit

@njit
def one_energy_vector(arr, ix, nmax):
    row = arr[ix]
    right = np.roll(row, -1)
    left = np.roll(row, 1)
    up = arr[(ix - 1) % nmax]
    down = arr[(ix + 1) % nmax]
    e = 0.5 * ((1 - 3 * np.cos(row - right)**2)
             + (1 - 3 * np.cos(row - left)**2)
             + (1 - 3 * np.cos(row - up)**2)
             + (1 - 3 * np.cos(row - down)**2))
    return e

@njit
def all_energy(arr, nmax):
    e = 0.0
    for i in range(nmax):
        e += np.sum(one_energy_vector(arr, i, nmax))
    return e

@njit
def get_order(arr, nmax):
    cx, sx = np.cos(arr), np.sin(arr)
    Qxx = np.mean(3 * cx * cx - 1)
    Qyy = np.mean(3 * sx * sx - 1)
    Qxy = np.mean(3 * cx * sx)
    Q = np.array([[Qxx, Qxy], [Qxy, Qyy]])
    vals = np.linalg.eigvals(Q)
    return np.max(vals)

@njit
def MC_step(arr, Ts, nmax):
    scale = 0.1 + Ts
    accept = 0
    for ix in range(nmax):
        for iy in range(nmax):
            old_angle = arr[ix, iy]
            e0 = one_energy_vector(arr, ix, nmax)[iy]
            arr[ix, iy] = old_angle + np.random.normal(0.0, scale)
            e1 = one_energy_vector(arr, ix, nmax)[iy]
            dE = e1 - e0
            if dE <= 0 or np.random.random() < np.exp(-dE / Ts):
                accept += 1
            else:
                arr[ix, iy] = old_angle
    return accept / (nmax * nmax)

def initdat(nmax):
    return np.random.random_sample((nmax, nmax)) * 2.0 * np.pi

def plotdat(arr, pflag, nmax):
    if pflag == 0:
        return
    u, v = np.cos(arr), np.sin(arr)
    x, y = np.arange(nmax), np.arange(nmax)
    if pflag == 1:
        mpl.rc('image', cmap='rainbow')
        cols = np.zeros((nmax, nmax))
        for i in range(nmax):
            cols[i] = one_energy_vector(arr, i, nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag == 2:
        mpl.rc('image', cmap='hsv')
        cols = arr % np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)
    quiveropts = dict(headlength=0, pivot='middle', headwidth=1, scale=1.1*nmax)
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
