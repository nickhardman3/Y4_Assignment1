import numpy as np
from mpi4py import MPI
from numba import njit

def _decompose_rows(nmax, size, rank):
    base = nmax // size
    extra = nmax % size
    if rank < extra:
        rows = base + 1
        start = rank * rows
    else:
        rows = base
        start = rank * base + extra
    return start, rows

def _halo_exchange_nb(local, comm, rank, size):
    up = (rank - 1 + size) % size
    dn = (rank + 1) % size
    reqs = [
        comm.Isend([local[1, :], MPI.DOUBLE], dest=up, tag=11),
        comm.Irecv([local[0, :], MPI.DOUBLE], source=up, tag=22),
        comm.Isend([local[-2, :], MPI.DOUBLE], dest=dn, tag=22),
        comm.Irecv([local[-1, :], MPI.DOUBLE], source=dn, tag=11),
    ]
    return reqs

def _halo_exchange(local, nmax, comm, rank, size):
    MPI.Request.Waitall(_halo_exchange_nb(local, comm, rank, size))


@njit(cache=True, fastmath=True) #numba-compiled version of single-site energy calculation
def _one_energy_nb(a, ix, iy, nmax):
    ixp = ix + 1
    ixm = ix - 1
    iyp = (iy + 1) % nmax
    iym = (iy - 1 + nmax) % nmax
    en = 0.0
    ang = a[ix, iy] - a[ixp, iy]; c = np.cos(ang); en += 0.5 * (1.0 - 3.0 * c * c)
    ang = a[ix, iy] - a[ixm, iy]; c = np.cos(ang); en += 0.5 * (1.0 - 3.0 * c * c)
    ang = a[ix, iy] - a[ix, iyp]; c = np.cos(ang); en += 0.5 * (1.0 - 3.0 * c * c)
    ang = a[ix, iy] - a[ix, iym]; c = np.cos(ang); en += 0.5 * (1.0 - 3.0 * c * c)
    return en

@njit(cache=True, fastmath=True)
def _energy_local(a, nmax, rows):
    s = 0.0
    for i in range(1, rows + 1):
        for j in range(nmax):
            s += _one_energy_nb(a, i, j, nmax)
    return s

@njit(cache=True, fastmath=True)
def _order_local(a, nmax, rows):
    q00 = 0.0
    q11 = 0.0
    q01 = 0.0
    N2 = float(nmax * nmax)
    for i in range(1, rows + 1):
        for j in range(nmax):
            c = np.cos(a[i, j])
            s = np.sin(a[i, j])
            q00 += 3.0 * c * c - 1.0
            q11 += 3.0 * s * s - 1.0
            q01 += 3.0 * c * s
    scale = 1.0 / (2.0 * N2)
    return q00 * scale, q11 * scale, q01 * scale

@njit(cache=True, fastmath=True)
def _mc_half_sweep_interior(a, angles, uniforms, nmax, rows, start_row, end_row, parity, g_start, Ts):
    acc = 0
    for i_local in range(start_row, end_row + 1):
        g_i = g_start + (i_local - 1)
        j_start = (parity - (g_i & 1)) & 1
        for j in range(j_start, nmax, 2):
            ang = angles[i_local - 1, j]
            u = uniforms[i_local - 1, j]
            en0 = _one_energy_nb(a, i_local, j, nmax)
            a[i_local, j] += ang
            en1 = _one_energy_nb(a, i_local, j, nmax)
            if en1 <= en0:
                acc += 1
            else:
                boltz = np.exp(-(en1 - en0) / Ts)
                if boltz >= u:
                    acc += 1
                else:
                    a[i_local, j] -= ang
    return acc


def one_energy(arr, ix, iy, nmax):
    return _one_energy_nb(arr, ix, iy, nmax)

def all_energy(arr_global_like, nmax):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    start, rows = _decompose_rows(nmax, comm.Get_size(), rank)
    local = _STATE['local']
    _halo_exchange(local, nmax, comm, rank, comm.Get_size())
    s = _energy_local(local, nmax, rows)
    return comm.allreduce(s, op=MPI.SUM)

def get_order(arr_global_like, nmax):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    start, rows = _decompose_rows(nmax, comm.Get_size(), rank)
    local = _STATE['local']
    _halo_exchange(local, nmax, comm, rank, comm.Get_size())
    q00_l, q11_l, q01_l = _order_local(local, nmax, rows)
    vec = np.array([q00_l, q11_l, q01_l], dtype=np.float64)
    comm.Allreduce(MPI.IN_PLACE, vec, op=MPI.SUM)  
    Q = np.array([[vec[0], vec[2], 0.0],
                  [vec[2], vec[1], 0.0],
                  [0.0,    0.0,   -0.5]])
    vals = np.linalg.eigvals(Q).real
    return float(np.max(vals))

def MC_step(arr_global_like, Ts, nmax):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    g_start, rows = _decompose_rows(nmax, size, rank)
    local = _STATE['local']
    scale = 0.1 + Ts

    angles = np.random.normal(0.0, scale, size=(rows, nmax))
    uniforms = np.random.random(size=(rows, nmax))

    accept_local = 0
    for parity in (0, 1):
        reqs = _halo_exchange_nb(local, comm, rank, size)
        if rows >= 3:
            accept_local += _mc_half_sweep_interior(local, angles, uniforms, nmax, rows, 2, rows - 1, parity, g_start, Ts)
        MPI.Request.Waitall(reqs)
        if rows >= 1:
            accept_local += _mc_half_sweep_interior(local, angles, uniforms, nmax, rows, 1, 1, parity, g_start, Ts)
        if rows >= 2:
            accept_local += _mc_half_sweep_interior(local, angles, uniforms, nmax, rows, rows, rows, parity, g_start, Ts)

    accept_global = comm.allreduce(accept_local, op=MPI.SUM)
    return accept_global / float(nmax * nmax)

def initdat(nmax):
    return np.random.random_sample((nmax, nmax)) * (2.0 * np.pi)

def plotdat(arr, pflag, nmax):
    if pflag == 0:
        return
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start, rows = _decompose_rows(nmax, size, rank)
    local = _STATE['local']
    sendbuf = local[1:rows+1, :].copy()
    counts = comm.gather(rows * nmax, root=0)
    if rank == 0:
        recvbuf = np.empty((nmax * nmax,), dtype=np.float64)
        displs = [0]
        for r in range(1, size):
            _, rr = _decompose_rows(nmax, size, r)
            displs.append(displs[-1] + counts[r-1])
    else:
        recvbuf = None; displs = None
    comm.Gatherv(sendbuf.ravel(), (recvbuf, counts, displs, MPI.DOUBLE), root=0)
    if rank == 0:
        full = recvbuf.reshape(nmax, nmax)
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        u = np.cos(full); v = np.sin(full)
        x = np.arange(nmax); y = np.arange(nmax)
        cols = np.zeros_like(full)
        if pflag == 1:
            mpl.rc('image', cmap='rainbow')
            for i in range(nmax):
                for j in range(nmax):
                    ip = (i + 1) % nmax; im = (i - 1 + nmax) % nmax
                    jp = (j + 1) % nmax; jm = (j - 1 + nmax) % nmax
                    en = 0.0
                    for (ii, jj) in [(ip, j), (im, j), (i, jp), (i, jm)]:
                        c = np.cos(full[i, j] - full[ii, jj])
                        en += 0.5 * (1.0 - 3.0 * c * c)
                    cols[i, j] = en
            norm = plt.Normalize(cols.min(), cols.max())
        elif pflag == 2:
            mpl.rc('image', cmap='hsv'); cols = full % np.pi
            norm = plt.Normalize(vmin=0, vmax=np.pi)
        else:
            mpl.rc('image', cmap='gist_gray'); cols = np.zeros_like(full)
            norm = plt.Normalize(vmin=0, vmax=1)
        quiveropts = dict(headlength=0, pivot='middle', headwidth=1, scale=1.1 * nmax)
        fig, ax = plt.subplots()
        ax.quiver(x, y, u, v, cols, norm=norm, **quiveropts)
        ax.set_aspect('equal'); plt.show()

def savedat(arr, nsteps, Ts, runtime, ratio, energy, order, nmax):
    if MPI.COMM_WORLD.Get_rank() != 0:
        return None
    import datetime
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
    return filename

def main(iterations, nmax, Ts, pflag):
    import time
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start, rows = _decompose_rows(nmax, size, rank)

    if rank == 0:
        global_lat = initdat(nmax).astype(np.float64, copy=False)
    else:
        global_lat = None

    counts = [(_decompose_rows(nmax, size, r)[1]) * nmax for r in range(size)]
    displs = np.zeros(size, dtype=np.int64)
    for r in range(1, size):
        displs[r] = displs[r-1] + counts[r-1]

    recv_flat = np.empty(rows * nmax, dtype=np.float64)
    if rank == 0:
        comm.Scatterv([global_lat.ravel(), counts, displs, MPI.DOUBLE], recv_flat, root=0)
    else:
        comm.Scatterv([None, counts, displs, MPI.DOUBLE], recv_flat, root=0)

    local = np.zeros((rows + 2, nmax), dtype=np.float64)
    local[1:rows+1, :] = recv_flat.reshape(rows, nmax)
    _STATE.clear(); _STATE['local'] = local

    plotdat(None, pflag, nmax)

    energy = np.zeros(iterations + 1, dtype=np.float64)
    ratio  = np.zeros(iterations + 1, dtype=np.float64)
    order  = np.zeros(iterations + 1, dtype=np.float64)

    energy[0] = all_energy(None, nmax)
    ratio[0]  = 0.5
    order[0]  = get_order(None, nmax)

    comm.Barrier()
    t0 = MPI.Wtime()
    for it in range(1, iterations + 1):
        ratio[it]  = MC_step(None, Ts, nmax)
        energy[it] = all_energy(None, nmax)
        order[it]  = get_order(None, nmax)
    comm.Barrier()
    runtime = MPI.Wtime() - t0

    if rank == 0:
        print(f"MPI: Size: {nmax}, Steps: {iterations}, T*: {Ts:.3f}: "
              f"Order: {order[iterations-1]:.3f}, Time: {runtime:.6f} s")

        savedat(None, iterations, Ts, runtime, ratio, energy, order, nmax)
        plotdat(None, pflag, nmax)

_STATE = {"local": None}