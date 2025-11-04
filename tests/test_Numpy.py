import numpy as np
import importlib
import pytest
import time

core = importlib.import_module("lebo.Core.core")
numpy_vec = importlib.import_module("lebo.Numpy.Numpy")
numpy_seq = importlib.import_module("lebo.Numpy.Numpy_Sequential")

@pytest.mark.parametrize("module", [numpy_vec, numpy_seq])
def test_all_energy_consistency_with_core(module):
    n = 10
    arr = np.random.RandomState(0).rand(n, n) * 2*np.pi
    e_core = core.all_energy(arr, n)
    e_mod = module.all_energy(arr, n)
    assert np.isclose(e_core, e_mod, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("module", [numpy_vec, numpy_seq])
def test_order_parameter_consistency_with_core(module):
    n = 10
    arr = np.random.RandomState(1).rand(n, n) * 2*np.pi
    s_core = core.get_order(arr, n)
    s_mod = module.get_order(arr, n)
    assert np.isclose(s_core, s_mod, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("module", [numpy_vec, numpy_seq])
def test_energy_rotation_invariance(module):
    n = 8
    arr = np.random.rand(n, n) * 2*np.pi
    e0 = module.all_energy(arr, n)
    e1 = module.all_energy(arr + 0.4, n)
    assert np.isclose(e0, e1, atol=1e-8)

@pytest.mark.parametrize("module", [numpy_vec, numpy_seq])
def test_order_parameter_range(module):
    n = 10
    arr = np.random.rand(n, n) * 2*np.pi
    s = module.get_order(arr, n)
    assert 0.0 <= s <= 1.0

@pytest.mark.parametrize("module", [numpy_vec, numpy_seq])
def test_equilibration_physical_trend(module):
    np.random.seed(0)
    n = 10
    theta = np.random.rand(n, n) * 2*np.pi
    for _ in range(300):
        module.MC_step(theta, 0.3, n)
    low_T = module.get_order(theta, n)

    theta = np.random.rand(n, n) * 2*np.pi
    for _ in range(300):
        module.MC_step(theta, 1.2, n)
    high_T = module.get_order(theta, n)

    assert low_T > high_T

def test_vector_vs_sequential_equivalence():
    n = 10
    arr = np.random.RandomState(2).rand(n, n) * 2*np.pi
    e_vec = numpy_vec.all_energy(arr, n)
    e_seq = numpy_seq.all_energy(arr, n)
    assert np.isclose(e_vec, e_seq, rtol=1e-5, atol=1e-5)

#@pytest.mark.parametrize("module", [numpy_vec, numpy_seq])
#@pytest.mark.benchmark(min_rounds=5)
#def test_monte_carlo_efficiency_numpy(benchmark, module):
#    n = 50
#    steps = 500
#    theta = np.random.rand(n, n) * 2 * np.pi
#
#    def run_simulation():
#        for _ in range(steps):
#            module.MC_step(theta, 0.6, n)
#
#    benchmark(run_simulation)