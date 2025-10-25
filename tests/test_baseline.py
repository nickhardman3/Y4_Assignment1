import numpy as np
import importlib

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

