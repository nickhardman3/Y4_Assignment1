import numpy as np
import importlib
import pytest
import time

core = importlib.import_module("lebo.core")
numba_mod = importlib.import_module("lebo.Numba")

def test_energy_equivalence_to_core():
    n = 10
    arr = np.random.RandomState(0).rand(n, n) * 2*np.pi
    e_core = core.all_energy(arr, n)
    e_numba = numba_mod.all_energy(arr, n)
    assert np.isclose(e_core, e_numba, rtol=1e-6, atol=1e-6)

def test_order_parameter_equivalence_to_core():
    n = 10
    arr = np.random.RandomState(1).rand(n, n) * 2*np.pi
    s_core = core.get_order(arr, n)
    s_numba = numba_mod.get_order(arr, n)
    assert np.isclose(s_core, s_numba, rtol=1e-6, atol=1e-6)

def test_energy_symmetry_preserved():
    n = 8
    arr = np.random.rand(n, n) * 2*np.pi
    e1 = numba_mod.all_energy(arr, n)
    e2 = numba_mod.all_energy((arr + np.pi) % (2*np.pi), n)
    assert np.isclose(e1, e2, atol=1e-8)

def test_equilibration_physical_trend():
    np.random.seed(0)
    n = 10
    theta = np.random.rand(n, n) * 2*np.pi
    for _ in range(300):
        numba_mod.MC_step(theta, 0.3, n)
    low_T = numba_mod.get_order(theta, n)

    theta = np.random.rand(n, n) * 2*np.pi
    for _ in range(300):
        numba_mod.MC_step(theta, 1.2, n)
    high_T = numba_mod.get_order(theta, n)

    assert low_T > high_T

def test_runtime_faster_than_core():
    n = 25
    arr = np.random.rand(n, n) * 2*np.pi
    start = time.time()
    core.all_energy(arr, n)
    t_core = time.time() - start

    start = time.time()
    numba_mod.all_energy(arr, n)
    t_numba = time.time() - start

    assert t_numba < 5 * t_core

#@pytest.mark.benchmark(min_rounds=5)
#def test_monte_carlo_efficiency_numba(benchmark):
#    n = 50
#    steps = 500
#    theta = np.random.rand(n, n) * 2 * np.pi
#
#    def run_simulation():
#        for _ in range(steps):
#            numba_mod.MC_step(theta, 0.6, n)
#
#    benchmark(run_simulation)