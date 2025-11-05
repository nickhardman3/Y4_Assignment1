import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def one_energy(arr, ix, iy, nmax): #vectorised neighbour selection using NumPy array operations to compute energy more efficiently
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax
    neighbors = np.array([
        arr[ixp, iy],
        arr[ixm, iy],
        arr[ix, iyp],
        arr[ix, iym]
    ])
    ang = arr[ix, iy] - neighbors
    en = 0.5 * np.sum(1.0 - 3.0 * np.cos(ang) ** 2)
    return en


def all_energy(arr, nmax): #fully vectorised energy calculation using np.roll to shift lattice and avoid explicit loops
    ixp = np.roll(arr, -1, axis=0)
    ixm = np.roll(arr, 1, axis=0)
    iyp = np.roll(arr, -1, axis=1)
    iym = np.roll(arr, 1, axis=1)
    en = 0.5 * (1.0 - 3.0 * np.cos(arr - ixp) ** 2)
    en += 0.5 * (1.0 - 3.0 * np.cos(arr - ixm) ** 2)
    en += 0.5 * (1.0 - 3.0 * np.cos(arr - iyp) ** 2)
    en += 0.5 * (1.0 - 3.0 * np.cos(arr - iym) ** 2)
    return np.sum(en)


def get_order(arr, nmax): #simplified Q-tensor computation using NumPy broadcasting instead of nested loops
    lab = np.zeros((3, nmax, nmax))
    lab[0] = np.cos(arr)
    lab[1] = np.sin(arr)
    Qab = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            Qab[a, b] = np.sum(3 * lab[a] * lab[b] - (a == b))
    Qab = Qab / (2 * nmax * nmax)
    vals = np.linalg.eigvals(Qab)
    return np.max(vals.real)


def MC_step(arr, Ts, nmax): #same algorithm as core but uses pre-generated random arrays and NumPy operations for sampling and energy tests
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
            arr[ix, iy] += ang
            en1 = one_energy(arr, ix, iy, nmax)
            if en1 <= en0:
                accept += 1
            else:
                boltz = np.exp(-(en1 - en0) / Ts)
                if boltz >= np.random.random():
                    accept += 1
                else:
                    arr[ix, iy] -= ang
    return accept / (nmax * nmax)


def initdat(nmax):
    return np.random.random_sample((nmax, nmax)) * 2.0 * np.pi


def plotdat(arr, pflag, nmax): #vectorised computation of per-site energy via np.vectorize for faster visualisation
    if pflag == 0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax, nmax))
    if pflag == 1:
        mpl.rc('image', cmap='rainbow')
        cols = np.vectorize(lambda i, j: one_energy(arr, i, j, nmax))(np.arange(nmax)[:, None], np.arange(nmax))
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
