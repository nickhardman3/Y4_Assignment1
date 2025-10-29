import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

# ============================================================

def all_energy(arr, nmax):
    dxp = np.roll(arr, -1, axis=0) - arr
    dxm = np.roll(arr, 1, axis=0) - arr
    dyp = np.roll(arr, -1, axis=1) - arr
    dym = np.roll(arr, 1, axis=1) - arr
    en = 0.5 * ((1 - 3*np.cos(dxp)**2)
              + (1 - 3*np.cos(dxm)**2)
              + (1 - 3*np.cos(dyp)**2)
              + (1 - 3*np.cos(dym)**2))
    return np.sum(en)

def one_energy_vector(arr, ix, nmax):
    row = arr[ix]
    ang_right = row - arr[(ix + 1) % nmax]
    ang_left  = row - arr[(ix - 1) % nmax]
    ang_up    = row - np.roll(row, 1)
    ang_down  = row - np.roll(row, -1)
    en = 0.5 * ((1.0 - 3.0 * np.cos(ang_right)**2)
              + (1.0 - 3.0 * np.cos(ang_left)**2)
              + (1.0 - 3.0 * np.cos(ang_up)**2)
              + (1.0 - 3.0 * np.cos(ang_down)**2))
    return en

def get_order(arr, nmax):
    lab = np.array([np.cos(arr), np.sin(arr), np.zeros_like(arr)])
    Qab = np.einsum('aij,bij->ab', lab, lab)
    delta = np.eye(3)
    Qab = (3 * Qab - delta * nmax**2) / (2 * nmax**2)
    vals, _ = np.linalg.eig(Qab)
    return vals.max()

# ============================================================

def MC_step(arr, Ts, nmax):
    scale = 0.1 + Ts
    accept = 0
    xran = np.random.randint(0, nmax, size=nmax*nmax)
    yran = np.random.randint(0, nmax, size=nmax*nmax)
    aran = np.random.normal(scale=scale, size=nmax*nmax)
    for i in range(nmax * nmax):
        ix, iy, ang = xran[i], yran[i], aran[i]
        e0 = one_energy_vector(arr, ix, nmax)[iy]
        arr[ix, iy] += ang
        e1 = one_energy_vector(arr, ix, nmax)[iy]
        dE = e1 - e0
        if dE <= 0 or np.exp(-dE / Ts) >= np.random.rand():
            accept += 1
        else:
            arr[ix, iy] -= ang
    return accept / (nmax * nmax)

# ============================================================

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
