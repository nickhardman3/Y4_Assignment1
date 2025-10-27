import numpy as np
import importlib
import time

ll = importlib.import_module("lebo.core")

def test_one_energy_periodicity():
    n = 5
    arr = np.random.RandomState(0).rand(n, n) * 2*np.pi
    e0 = ll.one_energy(arr, 2, 3, n)
    e1 = ll.one_energy(arr + 2*np.pi, 2, 3, n)
    assert np.isclose(e0, e1, atol=1e-12)

def test_all_energy_rotation_invariance():
    n = 6
    arr = np.random.RandomState(1).rand(n, n) * 2*np.pi
    e0 = ll.all_energy(arr, n)
    e1 = ll.all_energy(arr + 0.45, n)
    assert np.isclose(e0, e1, atol=1e-12)

def test_get_order_alignment_extremes():
    n = 10
    zeros = np.zeros((n, n))
    assert 0.9 <= ll.get_order(zeros, n) <= 1.0

def test_monte_carlo_energy_change_sign():
    n = 10
    theta = np.random.rand(n, n) * 2 * np.pi
    e_before = ll.all_energy(theta, n)
    theta_new = ll.monte_carlo(theta.copy(), n, tstar=0.7, nsteps=50)
    e_after = ll.all_energy(theta_new, n)
    assert not np.isclose(e_before, e_after, atol=1e-8)

def test_monte_carlo_efficiency():
    n = 20
    theta = np.random.rand(n, n) * 2 * np.pi
    start = time.time()
    ll.monte_carlo(theta, n, tstar=0.6, nsteps=200)
    duration = time.time() - start
    assert duration < 2.0

def test_order_parameter_range():
    n = 12
    theta = np.random.rand(n, n) * 2 * np.pi
    s = ll.get_order(theta, n)
    assert 0.0 <= s <= 1.0

def test_energy_symmetry():
    n = 8
    theta = np.random.rand(n, n) * 2 * np.pi
    e1 = ll.all_energy(theta, n)
    e2 = ll.all_energy((theta + np.pi) % (2*np.pi), n)
    assert np.isclose(e1, e2, atol=1e-8)

def test_equilibration_trend():
    n = 10
    theta = np.random.rand(n, n) * 2 * np.pi
    order_lowT = np.mean([ll.get_order(ll.monte_carlo(theta.copy(), n, tstar=0.3, nsteps=100), n) for _ in range(3)])
    order_highT = np.mean([ll.get_order(ll.monte_carlo(theta.copy(), n, tstar=1.2, nsteps=100), n) for _ in range(3)])
    assert order_lowT > order_highT