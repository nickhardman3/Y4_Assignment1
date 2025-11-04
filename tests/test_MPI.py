import numpy as np
import importlib
import pytest
try:
    from mpi4py import MPI
except Exception:
    pytest.skip("MPI not available; skipping MPI tests on CI", allow_module_level=True)

core = importlib.import_module("lebo.Core.core")
mpi_mod = importlib.import_module("lebo.MPI.MPI")
mpi_numba = importlib.import_module("lebo.MPI.MPI_Numba")

@pytest.mark.parametrize("module", [mpi_mod, mpi_numba])
def test_all_energy_consistency_with_core(module):
    n = 10
    arr = np.random.RandomState(0).rand(n, n) * 2*np.pi
    # Initialise fake MPI state for standalone test
    module._STATE["local"] = np.pad(arr, ((1,1),(0,0)), mode="wrap")
    e_core = core.all_energy(arr, n)
    e_mod = module.all_energy(arr, n)
    assert np.isclose(e_core, e_mod, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("module", [mpi_mod, mpi_numba])
def test_order_parameter_consistency_with_core(module):
    n = 10
    arr = np.random.RandomState(1).rand(n, n) * 2*np.pi
    module._STATE["local"] = np.pad(arr, ((1,1),(0,0)), mode="wrap")
    s_core = core.get_order(arr, n)
    s_mod = module.get_order(arr, n)
    assert np.isclose(s_core, s_mod, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("module", [mpi_mod, mpi_numba])
def test_energy_rotation_invariance(module):
    n = 8
    arr = np.random.rand(n, n) * 2*np.pi
    module._STATE["local"] = np.pad(arr, ((1,1),(0,0)), mode="wrap")
    e0 = module.all_energy(arr, n)
    e1 = module.all_energy(arr + 0.5, n)
    assert np.isclose(e0, e1, atol=1e-8)

@pytest.mark.parametrize("module", [mpi_mod, mpi_numba])
def test_order_parameter_range(module):
    n = 10
    arr = np.random.rand(n, n) * 2*np.pi
    module._STATE["local"] = np.pad(arr, ((1,1),(0,0)), mode="wrap")
    s = module.get_order(arr, n)
    assert 0.0 <= s <= 1.0
